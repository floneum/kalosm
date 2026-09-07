//! The frontend's broadcasting layer. **The IR has no implicit
//! broadcasting**: this module resolves a binary op's operand shapes and emits
//! the stride-0 `Restride` that `verify_l0` requires.
//!
//! The broadcast rule lives in [`fusor_ir::shape::broadcast_specs`] /
//! [`fusor_ir::shape::broadcast_shapes`]. Right-aligned: a source dim is
//! consumed when it equals the target or is 1 (stride 0); unmatched target dims
//! are inserted with stride 0 at **any** position; an unconsumed source dim is
//! an error.

use fusor_ir::ir::logical::Logical;
use fusor_ir::shape::{Dim, Dims, broadcast_shapes, broadcast_specs};

use crate::Result;
use crate::tensor::Tensor;

impl Tensor {
    /// Lift this value to `target` by emitting one stride-0 `Restride`.
    ///
    /// Always emits a node, even when the shapes already agree.
    pub fn broadcast_as(&self, target: &[Dim]) -> Result<Tensor> {
        let src = self.shape();
        let specs = broadcast_specs(&src, target)?;
        let bounds = crate::ops::view::bounds_for(&specs, &src);
        self.emit_here(Logical::Restride {
            specs,
            bounds,
            x: self.id,
        })
    }

    /// Alias of [`Tensor::broadcast_as`], preserved for source compatibility.
    pub fn expand(&self, target: &[Dim]) -> Result<Tensor> {
        self.broadcast_as(target)
    }
}

/// Lift both operands to their common shape and report it.
pub(crate) fn broadcast_pair(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor, Dims)> {
    let out = broadcast_shapes(&a.shape(), &b.shape())?;
    let ba = a.broadcast_as(&out)?;
    let bb = b.broadcast_as(&out)?;
    Ok((ba, bb, out))
}
