//! Comparison tensor operations: Eq, Lt, Lte, Gt, Gte
//! Returns 1.0/0.0 for floats or 1/0 for integers to match GPU backend semantics.

use pulp::Simd;

use crate::pairwise::SimdBinaryOp;
use crate::{ConcreteTensor, SimdElement, TensorBacking, materialize_expr};
use fusor_types::Layout;

// Comparison operation markers
macro_rules! define_cmp_marker {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Copy, Clone)]
            pub struct $name;
        )*
    };
}
define_cmp_marker!(EqOp, NeOp, LtOp, LteOp, GtOp, GteOp);

// Helper trait for types that can represent 0 and 1
trait NumericBool: SimdElement {
    fn zero() -> Self;
    fn one() -> Self;
}

impl NumericBool for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl NumericBool for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl NumericBool for i8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl NumericBool for i16 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl NumericBool for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl NumericBool for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl NumericBool for u8 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl NumericBool for u16 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl NumericBool for u32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl NumericBool for u64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

// Macro for scalar-only comparison ops (convert boolean mask to 1.0/0.0)
macro_rules! impl_scalar_comparison_op {
    ($op:ty, $cmp_fn:expr, $elem:ty) => {
        impl SimdBinaryOp<$elem> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                _simd: S,
                a: <$elem as SimdElement>::Simd<S>,
                b: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                // Process each lane with scalar comparison
                let lane_count = std::mem::size_of::<<$elem as SimdElement>::Simd<S>>()
                    / std::mem::size_of::<$elem>();
                let mut temp_out = [<$elem>::default(); crate::MAX_SIMD_LANES];

                // Safe: cast SIMD refs to scalar slices via bytemuck
                let slice_a: &[$elem] = pulp::bytemuck::cast_slice(std::slice::from_ref(&a));
                let slice_b: &[$elem] = pulp::bytemuck::cast_slice(std::slice::from_ref(&b));

                let cmp: fn($elem, $elem) -> bool = $cmp_fn;
                for i in 0..lane_count {
                    temp_out[i] = if cmp(slice_a[i], slice_b[i]) {
                        <$elem as NumericBool>::one()
                    } else {
                        <$elem as NumericBool>::zero()
                    };
                }

                // Safe: reconstruct SIMD from scalar slice via as_simd
                let (simd_slice, _) = <$elem as SimdElement>::as_simd::<S>(&temp_out[..lane_count]);
                simd_slice[0]
            }

            #[inline(always)]
            fn apply_scalar(a: $elem, b: $elem) -> $elem {
                let cmp: fn($elem, $elem) -> bool = $cmp_fn;
                if cmp(a, b) {
                    <$elem as NumericBool>::one()
                } else {
                    <$elem as NumericBool>::zero()
                }
            }
        }
    };
}

// Implement comparison ops for all numeric types
macro_rules! impl_all_comparisons {
    ($($elem:ty),*) => {
        $(
            impl_scalar_comparison_op!(EqOp, |a: $elem, b: $elem| a == b, $elem);
            impl_scalar_comparison_op!(NeOp, |a: $elem, b: $elem| a != b, $elem);
            impl_scalar_comparison_op!(LtOp, |a: $elem, b: $elem| a < b, $elem);
            impl_scalar_comparison_op!(LteOp, |a: $elem, b: $elem| a <= b, $elem);
            impl_scalar_comparison_op!(GtOp, |a: $elem, b: $elem| a > b, $elem);
            impl_scalar_comparison_op!(GteOp, |a: $elem, b: $elem| a >= b, $elem);
        )*
    };
}

impl_all_comparisons!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

/// Macro to define comparison tensor expression types
macro_rules! define_comparison_tensor_op {
    ($name:ident, $simd_op:ty) => {
        pub struct $name<
            E: SimdElement,
            const R: usize,
            T1: TensorBacking<R, Elem = E>,
            T2: TensorBacking<R, Elem = E>,
        > {
            lhs: T1,
            rhs: T2,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const R: usize, T1, T2> $name<E, R, T1, T2>
        where
            E: SimdElement,
            T1: TensorBacking<R, Elem = E>,
            T2: TensorBacking<R, Elem = E>,
        {
            pub fn new(lhs: T1, rhs: T2) -> Self {
                Self {
                    lhs,
                    rhs,
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<E, const R: usize, T1, T2> crate::LazyBacking for $name<E, R, T1, T2>
        where
            E: SimdElement + Default,
            $simd_op: SimdBinaryOp<E>,
            T1: TensorBacking<R, Elem = E>,
            T2: TensorBacking<R, Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op>::apply_scalar(self.lhs.eval_scalar(idx), self.rhs.eval_scalar(idx))
            }

            #[inline(always)]
            fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op>::apply_simd_vec(
                    simd,
                    self.lhs.eval_simd(simd, base_idx),
                    self.rhs.eval_simd(simd, base_idx),
                )
            }
        }

        impl<E, const R: usize, T1, T2> TensorBacking<R> for $name<E, R, T1, T2>
        where
            E: SimdElement + Default,
            $simd_op: SimdBinaryOp<E>,
            T1: TensorBacking<R, Elem = E>,
            T2: TensorBacking<R, Elem = E>,
        {
            fn layout(&self) -> Layout {
                Layout::contiguous(self.lhs.layout().shape())
            }

            fn to_concrete(&self) -> ConcreteTensor<E, R> {
                let shape: [usize; R] = self
                    .lhs
                    .layout()
                    .shape()
                    .try_into()
                    .expect("Shape length mismatch");
                materialize_expr(self, shape)
            }
        }
    };
}

// Comparison tensor expression types
define_comparison_tensor_op!(Eq, EqOp);
define_comparison_tensor_op!(Ne, NeOp);
define_comparison_tensor_op!(Lt, LtOp);
define_comparison_tensor_op!(Lte, LteOp);
define_comparison_tensor_op!(Gt, GtOp);
define_comparison_tensor_op!(Gte, GteOp);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq_scalar() {
        assert_eq!(EqOp::apply_scalar(1.0f32, 1.0f32), 1.0);
        assert_eq!(EqOp::apply_scalar(1.0f32, 2.0f32), 0.0);
    }

    #[test]
    fn test_lt_scalar() {
        assert_eq!(LtOp::apply_scalar(1.0f32, 2.0f32), 1.0);
        assert_eq!(LtOp::apply_scalar(2.0f32, 1.0f32), 0.0);
        assert_eq!(LtOp::apply_scalar(1.0f32, 1.0f32), 0.0);
    }
}
