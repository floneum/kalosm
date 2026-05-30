//! Scalar (tensor op scalar) operations: AddScalar, SubScalar, MulScalar, DivScalar

use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Sub as StdSub};

use pulp::Simd;

use crate::pairwise::{AddOp, DivOp, MulOp, SubOp};
use crate::{ConcreteTensor, SimdElement, TensorBacking, define_tensor_op, materialize_expr};
use fusor_types::Layout;

// Scalar tensor operations
define_tensor_op!(@scalar AddScalar, AddOp, std_trait = StdAdd);
define_tensor_op!(@scalar SubScalar, SubOp, std_trait = StdSub);
define_tensor_op!(@scalar MulScalar, MulOp, std_trait = StdMul);
define_tensor_op!(@scalar DivScalar, DivOp, std_trait = StdDiv);

/// A scalar value broadcasted to a tensor shape
pub struct Broadcast<E: SimdElement, const R: usize> {
    scalar: E,
    shape: [usize; R],
}

impl<E: SimdElement, const R: usize> Broadcast<E, R> {
    pub fn new(scalar: E, shape: [usize; R]) -> Self {
        Self { scalar, shape }
    }
}

impl<E: SimdElement + Default, const R: usize> crate::LazyBacking for Broadcast<E, R> {
    type Elem = E;

    #[inline(always)]
    fn eval_scalar(&self, _idx: usize) -> E {
        self.scalar
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, simd: S, _base_idx: usize) -> E::Simd<S> {
        E::splat(simd, self.scalar)
    }
}

impl<E: SimdElement + Default, const R: usize> TensorBacking<R> for Broadcast<E, R> {
    fn layout(&self) -> Layout {
        Layout::contiguous(&self.shape)
    }

    fn to_concrete(&self) -> ConcreteTensor<E, R> {
        materialize_expr(self, self.shape)
    }
}
