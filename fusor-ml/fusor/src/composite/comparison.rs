//! Comparison operations that work on both CPU and GPU backends.
//!
//! These operations return tensors with 1.0 for true and 0.0 for false.

use crate::cpu::{EqOp, GtOp, GteOp, LtOp, LteOp, NeOp, SimdBinaryOp};
use crate::gpu::DataType;
use crate::{SimdElement, Tensor};

/// Emit a scalar-comparison method that dispatches CPU/GPU.
///
/// `$method` is the public method name (e.g. `eq_scalar`).
/// `$op` is the SIMD op marker type used in the where-clause (e.g. `EqOp`).
/// `$cpu_method` is the matching method on the CPU tensor (e.g. `eq_scalar`).
/// `$gpu_call` is a closure body `|t, s|` that emits the GPU code path.
macro_rules! scalar_cmp {
    ($(#[$meta:meta])* $method:ident, $op:ident, $cpu_method:ident, |$gt:ident, $gs:ident| $gpu_call:expr) => {
        $(#[$meta])*
        pub fn $method(&self, scalar: D) -> Self
        where
            $op: SimdBinaryOp<D>,
        {
            self.dispatch_ref(
                |t| t.as_ref().$cpu_method(scalar).to_concrete(),
                |$gt| { let $gs = scalar; $gpu_call },
            )
        }
    };
}

/// Emit a tensor-tensor comparison method that runs on CPU only.
macro_rules! tensor_cmp {
    ($(#[$meta:meta])* $method:ident, $op:ident, $cpu_method:ident) => {
        $(#[$meta])*
        pub fn $method(&self, rhs: &Self) -> Self
        where
            $op: SimdBinaryOp<D>,
        {
            self.dispatch_cpu_only_pair(rhs, |a, b| a.as_ref().$cpu_method(b.as_ref()).to_concrete())
        }
    };
}

/// Emit a fusor-core compatible alias that forwards to `$target`.
macro_rules! cmp_alias {
    ($(#[$meta:meta])* $method:ident, $op:ident, $target:ident) => {
        $(#[$meta])*
        pub fn $method(&self, rhs: D) -> Self
        where
            $op: SimdBinaryOp<D>,
        {
            self.$target(rhs)
        }
    };
}

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + Default,
{
    tensor_cmp!(
        /// Element-wise equality comparison between two tensors.
        ///
        /// Returns 1.0 where elements are equal, 0.0 otherwise.
        /// Note: GPU comparison is only available for CPU tensors at this time.
        eq_tensor, EqOp, eq
    );

    tensor_cmp!(
        /// Element-wise inequality comparison between two tensors.
        ///
        /// Returns 1.0 where elements are not equal, 0.0 otherwise.
        /// Note: GPU comparison is only available for CPU tensors at this time.
        ne_tensor, NeOp, ne
    );

    tensor_cmp!(
        /// Element-wise less-than comparison between two tensors.
        ///
        /// Returns 1.0 where self < rhs, 0.0 otherwise.
        /// Note: GPU comparison is only available for CPU tensors at this time.
        lt_tensor, LtOp, lt
    );

    tensor_cmp!(
        /// Element-wise less-than-or-equal comparison between two tensors.
        ///
        /// Returns 1.0 where self <= rhs, 0.0 otherwise.
        /// Note: GPU comparison is only available for CPU tensors at this time.
        lte_tensor, LteOp, lte
    );

    tensor_cmp!(
        /// Element-wise greater-than comparison between two tensors.
        ///
        /// Returns 1.0 where self > rhs, 0.0 otherwise.
        /// Note: GPU comparison is only available for CPU tensors at this time.
        gt_tensor, GtOp, gt
    );

    tensor_cmp!(
        /// Element-wise greater-than-or-equal comparison between two tensors.
        ///
        /// Returns 1.0 where self >= rhs, 0.0 otherwise.
        /// Note: GPU comparison is only available for CPU tensors at this time.
        gte_tensor, GteOp, gte
    );

    scalar_cmp!(
        /// Element-wise equality comparison with a scalar.
        ///
        /// Returns 1.0 where elements equal the scalar, 0.0 otherwise.
        eq_scalar, EqOp, eq_scalar, |t, s| t.eq(s)
    );

    scalar_cmp!(
        /// Element-wise inequality comparison with a scalar.
        ///
        /// Returns 1.0 where elements are not equal to the scalar, 0.0 otherwise.
        ne_scalar, NeOp, ne_scalar, |t, s| {
            let eq: crate::gpu::Tensor<R, D> = t.eq(s);
            eq.eq(D::zero())
        }
    );

    scalar_cmp!(
        /// Element-wise less-than comparison with a scalar.
        ///
        /// Returns 1.0 where self < scalar, 0.0 otherwise.
        lt_scalar, LtOp, lt_scalar, |t, s| t.lt(s)
    );

    scalar_cmp!(
        /// Element-wise less-than-or-equal comparison with a scalar.
        ///
        /// Returns 1.0 where self <= scalar, 0.0 otherwise.
        lte_scalar, LteOp, lte_scalar, |t, s| t.lte(s)
    );

    scalar_cmp!(
        /// Element-wise greater-than comparison with a scalar.
        ///
        /// Returns 1.0 where self > scalar, 0.0 otherwise.
        gt_scalar, GtOp, gt_scalar, |t, s| t.mt(s)
    );

    scalar_cmp!(
        /// Element-wise greater-than-or-equal comparison with a scalar.
        ///
        /// Returns 1.0 where self >= scalar, 0.0 otherwise.
        gte_scalar, GteOp, gte_scalar, |t, s| t.mte(s)
    );

    cmp_alias!(
        /// Element-wise equality comparison with a scalar (fusor-core compatible API).
        ///
        /// Returns 1.0 where elements equal the scalar, 0.0 otherwise.
        /// This is an alias for `eq_scalar` to match fusor-core's API.
        eq, EqOp, eq_scalar
    );

    cmp_alias!(
        /// Element-wise inequality comparison with a scalar (fusor-core compatible API).
        ///
        /// Returns 1.0 where elements are not equal to the scalar, 0.0 otherwise.
        /// This is an alias for `ne_scalar`.
        ne, NeOp, ne_scalar
    );

    cmp_alias!(
        /// Element-wise less-than comparison with a scalar (fusor-core compatible API).
        ///
        /// Returns 1.0 where self < scalar, 0.0 otherwise.
        /// This is an alias for `lt_scalar` to match fusor-core's API.
        lt, LtOp, lt_scalar
    );

    cmp_alias!(
        /// Element-wise less-than-or-equal comparison with a scalar (fusor-core compatible API).
        ///
        /// Returns 1.0 where self <= scalar, 0.0 otherwise.
        /// This is an alias for `lte_scalar` to match fusor-core's API.
        lte, LteOp, lte_scalar
    );

    cmp_alias!(
        /// Element-wise greater-than comparison with a scalar (fusor-core compatible API).
        ///
        /// Returns 1.0 where self > scalar, 0.0 otherwise.
        /// This is an alias for `gt_scalar` to match fusor-core's API.
        /// Named `mt` (more than) to match fusor-core.
        mt, GtOp, gt_scalar
    );

    cmp_alias!(
        /// Element-wise greater-than-or-equal comparison with a scalar (fusor-core compatible API).
        ///
        /// Returns 1.0 where self >= scalar, 0.0 otherwise.
        /// This is an alias for `gte_scalar` to match fusor-core's API.
        /// Named `mte` (more than or equal) to match fusor-core.
        mte, GteOp, gte_scalar
    );
}
