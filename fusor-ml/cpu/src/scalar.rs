//! Scalar (tensor op scalar) operations: AddScalar, SubScalar, MulScalar, DivScalar

use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Sub as StdSub};

use pulp::Simd;

use crate::pairwise::{AddOp, DivOp, MulOp, SimdBinaryOp, SubOp};
use crate::{ConcreteTensor, SimdElement, TensorBacking, materialize_expr};
use fusor_types::Layout;

/// Macro to define scalar tensor operations (AddScalar, SubScalar, MulScalar, DivScalar)
macro_rules! define_scalar_tensor_op {
    ($name:ident, $std_trait:ident, $simd_op:ty) => {
        pub struct $name<E: SimdElement, const R: usize, T: TensorBacking<R, Elem = E>> {
            tensor: T,
            scalar: E,
        }

        impl<E, const R: usize, T> $name<E, R, T>
        where
            E: SimdElement,
            T: TensorBacking<R, Elem = E>,
        {
            pub fn new(tensor: T, scalar: E) -> Self {
                Self { tensor, scalar }
            }
        }

        impl<E, const R: usize, T> crate::LazyBacking for $name<E, R, T>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdBinaryOp<E>,
            T: TensorBacking<R, Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op>::apply_scalar(self.tensor.eval_scalar(idx), self.scalar)
            }

            #[inline(always)]
            fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op>::apply_simd_vec(
                    simd,
                    self.tensor.eval_simd(simd, base_idx),
                    E::splat(simd, self.scalar),
                )
            }
        }

        impl<E, const R: usize, T> TensorBacking<R> for $name<E, R, T>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdBinaryOp<E>,
            T: TensorBacking<R, Elem = E>,
        {
            fn layout(&self) -> Layout {
                Layout::contiguous(self.tensor.layout().shape())
            }

            fn to_concrete(&self) -> ConcreteTensor<E, R> {
                let shape: [usize; R] = self
                    .tensor
                    .layout()
                    .shape()
                    .try_into()
                    .expect("Shape length mismatch");
                materialize_expr(self, shape)
            }
        }
    };
}

// Scalar tensor operations
define_scalar_tensor_op!(AddScalar, StdAdd, AddOp);
define_scalar_tensor_op!(SubScalar, StdSub, SubOp);
define_scalar_tensor_op!(MulScalar, StdMul, MulOp);
define_scalar_tensor_op!(DivScalar, StdDiv, DivOp);

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
