//! Fixed-shape logical programs with packed storage and explicit state feedback.
//!
//! Logical read recipes prove workgroup ownership. Storage barriers order
//! work within a region; dispatch boundaries order communication between
//! workgroups. Small programs can use one region and one workgroup. A checked
//! arena packs nonoverlapping lifetimes, and explicit feedback commits state
//! after all old-state reads. Programs own their storage and compiled kernels.
mod emit;
mod features;
mod index;
mod plan;
mod regions;
mod run;
mod workloads;

pub use features::{MatrixInstructions, ProgramAcceleration};
pub use plan::{Input, Plan, ProgramOptions, ProgramStats, Uniform};
pub use run::Program;
