//! Composite operations that work on both CPU and GPU backends.
//!
//! These operations are built from primitive operations and work uniformly
//! across CPU and GPU tensors via the Tensor abstraction.

mod activations;
mod comparison;
mod construction;
mod conv;
mod flash_attention;
pub mod index;
mod index_select;
mod math;
mod normalization;
pub mod pool;
mod reductions;
mod rope;
mod shape;
mod to_vec;
mod where_cond;

pub use flash_attention::MaskKind;
pub use rope::{RopeCache, base_inverse_frequency};
pub use shape::{arange, arange_step, cat, stack};
pub use to_vec::{ToVec1, ToVec2, ToVec3};
