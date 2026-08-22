//! The primitive op surface. Every entry mints one Logical node; none of them
//! chooses a kernel, a layout or a tiling.

pub(crate) mod cast;
pub(crate) mod comparison;
pub(crate) mod elementwise;
pub(crate) mod index;
pub(crate) mod matmul;
pub(crate) mod reduce;
pub(crate) mod scalar_arith;
pub(crate) mod view;
