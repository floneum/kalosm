//! Comparison tensor operations: Eq, Lt, Lte, Gt, Gte
//! Returns 1.0/0.0 for floats or 1/0 for integers to match GPU backend semantics.

use pulp::Simd;

use crate::pairwise::SimdBinaryOp;
use crate::{SimdElement, define_tensor_op};

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

// Comparison tensor expression types
define_tensor_op!(@binary Eq, EqOp);
define_tensor_op!(@binary Ne, NeOp);
define_tensor_op!(@binary Lt, LtOp);
define_tensor_op!(@binary Lte, LteOp);
define_tensor_op!(@binary Gt, GtOp);
define_tensor_op!(@binary Gte, GteOp);
