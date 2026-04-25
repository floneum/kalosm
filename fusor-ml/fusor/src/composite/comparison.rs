//! Comparison operations that work on both CPU and GPU backends.
//!
//! These operations return tensors with 1.0 for true and 0.0 for false.

use crate::{SimdElement, Tensor};
use fusor_core::DataType;
use fusor_cpu::{EqOp, GtOp, GteOp, LtOp, LteOp, NeOp, SimdBinaryOp};

impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: fusor_cpu::TensorBacking<R, Elem = D>,
{
    /// Element-wise equality comparison between two tensors.
    ///
    /// Returns 1.0 where elements are equal, 0.0 otherwise.
    pub fn eq_tensor(&self, rhs: &Self) -> Tensor<R, D>
    where
        EqOp: SimdBinaryOp<D>,
    {
        self.dispatch_pair_concrete(
            rhs,
            |a, b| a.as_ref().eq(b.as_ref()).to_concrete(),
            |a, b| a.eq_tensor(b),
        )
    }

    /// Element-wise inequality comparison between two tensors.
    ///
    /// Returns 1.0 where elements are not equal, 0.0 otherwise.
    /// Note: GPU comparison is only available for CPU tensors at this time.
    pub fn ne_tensor(&self, rhs: &Self) -> Tensor<R, D>
    where
        NeOp: SimdBinaryOp<D>,
    {
        self.dispatch_cpu_only_pair(rhs, |a, b| a.as_ref().ne(b.as_ref()).to_concrete())
    }

    /// Element-wise less-than comparison between two tensors.
    ///
    /// Returns 1.0 where self < rhs, 0.0 otherwise.
    pub fn lt_tensor(&self, rhs: &Self) -> Tensor<R, D>
    where
        LtOp: SimdBinaryOp<D>,
    {
        self.dispatch_pair_concrete(
            rhs,
            |a, b| a.as_ref().lt(b.as_ref()).to_concrete(),
            |a, b| a.lt_tensor(b),
        )
    }

    /// Element-wise less-than-or-equal comparison between two tensors.
    ///
    /// Returns 1.0 where self <= rhs, 0.0 otherwise.
    pub fn lte_tensor(&self, rhs: &Self) -> Tensor<R, D>
    where
        LteOp: SimdBinaryOp<D>,
    {
        self.dispatch_pair_concrete(
            rhs,
            |a, b| a.as_ref().lte(b.as_ref()).to_concrete(),
            |a, b| a.lte_tensor(b),
        )
    }

    /// Element-wise greater-than comparison between two tensors.
    ///
    /// Returns 1.0 where self > rhs, 0.0 otherwise.
    pub fn gt_tensor(&self, rhs: &Self) -> Tensor<R, D>
    where
        GtOp: SimdBinaryOp<D>,
    {
        self.dispatch_pair_concrete(
            rhs,
            |a, b| a.as_ref().gt(b.as_ref()).to_concrete(),
            |a, b| a.gt_tensor(b),
        )
    }

    /// Element-wise greater-than-or-equal comparison between two tensors.
    ///
    /// Returns 1.0 where self >= rhs, 0.0 otherwise.
    pub fn gte_tensor(&self, rhs: &Self) -> Tensor<R, D>
    where
        GteOp: SimdBinaryOp<D>,
    {
        self.dispatch_pair_concrete(
            rhs,
            |a, b| a.as_ref().gte(b.as_ref()).to_concrete(),
            |a, b| a.gte_tensor(b),
        )
    }

    /// Element-wise equality comparison with a scalar.
    ///
    /// Returns 1.0 where elements equal the scalar, 0.0 otherwise.
    pub fn eq_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        EqOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().eq_scalar(scalar).to_concrete(),
            |t| t.eq(scalar),
        )
    }

    /// Element-wise inequality comparison with a scalar.
    ///
    /// Returns 1.0 where elements are not equal to the scalar, 0.0 otherwise.
    pub fn ne_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        NeOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().ne_scalar(scalar).to_concrete(),
            |t| {
                let eq: fusor_core::Tensor<R, D> = t.eq(scalar);
                eq.eq(D::zero())
            },
        )
    }

    /// Element-wise less-than comparison with a scalar.
    ///
    /// Returns 1.0 where self < scalar, 0.0 otherwise.
    pub fn lt_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        LtOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().lt_scalar(scalar).to_concrete(),
            |t| t.lt(scalar),
        )
    }

    /// Element-wise less-than-or-equal comparison with a scalar.
    ///
    /// Returns 1.0 where self <= scalar, 0.0 otherwise.
    pub fn lte_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        LteOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().lte_scalar(scalar).to_concrete(),
            |t| t.lte(scalar),
        )
    }

    /// Element-wise greater-than comparison with a scalar.
    ///
    /// Returns 1.0 where self > scalar, 0.0 otherwise.
    pub fn gt_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        GtOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().gt_scalar(scalar).to_concrete(),
            |t| t.mt(scalar),
        )
    }

    /// Element-wise greater-than-or-equal comparison with a scalar.
    ///
    /// Returns 1.0 where self >= scalar, 0.0 otherwise.
    pub fn gte_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        GteOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().gte_scalar(scalar).to_concrete(),
            |t| t.mte(scalar),
        )
    }

    /// Element-wise equality comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where elements equal the scalar, 0.0 otherwise.
    /// This is an alias for `eq_scalar` to match fusor-core's API.
    pub fn eq(&self, rhs: D) -> Tensor<R, D>
    where
        EqOp: SimdBinaryOp<D>,
    {
        self.eq_scalar(rhs)
    }

    /// Element-wise inequality comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where elements are not equal to the scalar, 0.0 otherwise.
    /// This is an alias for `ne_scalar`.
    pub fn ne(&self, rhs: D) -> Tensor<R, D>
    where
        NeOp: SimdBinaryOp<D>,
    {
        self.ne_scalar(rhs)
    }

    /// Element-wise less-than comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where self < scalar, 0.0 otherwise.
    /// This is an alias for `lt_scalar` to match fusor-core's API.
    pub fn lt(&self, rhs: D) -> Tensor<R, D>
    where
        LtOp: SimdBinaryOp<D>,
    {
        self.lt_scalar(rhs)
    }

    /// Element-wise less-than-or-equal comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where self <= scalar, 0.0 otherwise.
    /// This is an alias for `lte_scalar` to match fusor-core's API.
    pub fn lte(&self, rhs: D) -> Tensor<R, D>
    where
        LteOp: SimdBinaryOp<D>,
    {
        self.lte_scalar(rhs)
    }

    /// Element-wise greater-than comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where self > scalar, 0.0 otherwise.
    /// This is an alias for `gt_scalar` to match fusor-core's API.
    /// Named `mt` (more than) to match fusor-core.
    pub fn mt(&self, rhs: D) -> Tensor<R, D>
    where
        GtOp: SimdBinaryOp<D>,
    {
        self.gt_scalar(rhs)
    }

    /// Element-wise greater-than-or-equal comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where self >= scalar, 0.0 otherwise.
    /// This is an alias for `gte_scalar` to match fusor-core's API.
    /// Named `mte` (more than or equal) to match fusor-core.
    pub fn mte(&self, rhs: D) -> Tensor<R, D>
    where
        GteOp: SimdBinaryOp<D>,
    {
        self.gte_scalar(rhs)
    }
}
