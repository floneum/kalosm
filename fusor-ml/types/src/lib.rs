//! Common types for fusor tensor libraries
//!
//! This crate provides shared types used by both `fusor-core` (GPU) and `fusor-cpu` (CPU) crates.

mod into_tensor;
mod layout;
mod rank;
mod tensor_slice;

pub use into_tensor::*;
pub use layout::*;
pub use rank::*;
pub use tensor_slice::*;
