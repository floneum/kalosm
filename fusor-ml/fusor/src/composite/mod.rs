//! Composite operations that work on both CPU and GPU backends.
//!
//! These operations are built from primitive operations and work uniformly
//! across CPU and GPU tensors via the Tensor abstraction.

mod activations;
mod attention;
mod comparison;
mod construction;
mod conv;
pub mod index;
mod index_select;
mod math;
mod normalization;
pub mod pool;
mod reductions;
mod rope;
mod shape;
mod to_vec;
mod upsample;
mod where_cond;

pub use attention::MaskKind;
pub use rope::{RopeCache, base_inverse_frequency};
pub(crate) use shape::broadcast_shapes;
pub use shape::{arange, arange_step, cat, stack};
pub use to_vec::ToVec;
