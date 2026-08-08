//! The primitive op surface. Every entry mints one L0 node; none of them
//! chooses a kernel, a layout or a tiling.

pub mod alias;
pub mod cast;
pub mod comparison;
pub mod elementwise;
pub mod index;
pub mod matmul;
pub mod reduce;
pub mod scalar_arith;
pub mod view;
