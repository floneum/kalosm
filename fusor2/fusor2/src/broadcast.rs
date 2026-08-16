//! The frontend's broadcasting layer. **The IR has no implicit
//! broadcasting**: this module resolves a binary op's operand shapes and emits
//! the stride-0 `Restride` that `verify_l0` requires.
//!
//! The broadcast rule lives in [`fusor2_ir::shape::broadcast_specs`] /
//! [`fusor2_ir::shape::broadcast_shapes`]. Right-aligned: a source dim is
//! consumed when it equals the target or is 1 (stride 0); unmatched target dims
//! are inserted with stride 0 at **any** position; an unconsumed source dim is
//! an error.

use fusor2_ir::ir::logical::Logical;
use fusor2_ir::shape::{Dim, Dims, broadcast_shapes, broadcast_specs};

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

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::shape::StrideSpec;

    fn dims(v: &[u64]) -> Vec<Dim> {
        v.iter().map(|&d| Dim::Const(d)).collect()
    }

    #[test]
    fn right_aligned_specs() {
        let s = broadcast_specs(&dims(&[3]), &dims(&[2, 3])).unwrap();
        assert_eq!(
            &s[..],
            &[
                StrideSpec::broadcast(Dim::Const(2)),
                StrideSpec::dim(0, Dim::Const(3)),
            ]
        );
    }

    #[test]
    fn middle_axis_becomes_multiplier_zero() {
        let s = broadcast_specs(&dims(&[2, 1, 4]), &dims(&[2, 3, 4])).unwrap();
        assert_eq!(s[1].multiplier, 0);
        assert_eq!(s[0].multiplier, 1);
        assert_eq!(s[2].multiplier, 1);
    }

    #[test]
    fn unconsumed_source_dim_is_an_error() {
        assert!(broadcast_specs(&dims(&[5]), &dims(&[2, 3])).is_err());
    }

    #[test]
    fn unmatched_target_dim_may_be_inserted_anywhere() {
        // [2, 4] into [2, 3, 4]: the 3 is inserted at position 1, not on the
        // left, so the leading 2 still matches the source's leading 2.
        let s = broadcast_specs(&dims(&[2, 4]), &dims(&[2, 3, 4])).unwrap();
        assert_eq!((s[0].multiplier, s[0].input_dim), (1, 0));
        assert_eq!(s[1].multiplier, 0);
        assert_eq!((s[2].multiplier, s[2].input_dim), (1, 1));
    }

    #[test]
    fn rank_zero_broadcasts_into_anything() {
        let s = broadcast_specs(&[], &dims(&[2, 3])).unwrap();
        assert!(s.iter().all(|x| x.multiplier == 0));
    }

    #[test]
    fn broadcasting_a_shape_to_itself_is_the_identity_view() {
        let shape = dims(&[2, 3]);
        let s = broadcast_specs(&shape, &shape).unwrap();
        assert_eq!(
            &s[..],
            &[
                StrideSpec::dim(0, Dim::Const(2)),
                StrideSpec::dim(1, Dim::Const(3)),
            ]
        );
    }
}
