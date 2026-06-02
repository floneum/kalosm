//! Pairwise (binary) tensor operations: Add, Sub, Mul, Div

use std::ops::{Add as StdAdd, Div as StdDiv, Mul as StdMul, Rem as StdRem, Sub as StdSub};

use pulp::Simd;

use crate::SimdElement;
use crate::define_tensor_op;

/// Trait for binary operations that have SIMD support
pub trait SimdBinaryOp<E: SimdElement>: Copy {
    /// Apply operation to SIMD vectors
    fn apply_simd_vec<S: Simd>(simd: S, a: E::Simd<S>, b: E::Simd<S>) -> E::Simd<S>;

    /// Apply operation to scalars
    fn apply_scalar(a: E, b: E) -> E;
}

// Operation marker macro and definitions
macro_rules! define_op_marker {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Copy, Clone)]
            pub struct $name;
        )*
    };
}
define_op_marker!(AddOp, SubOp, MulOp, DivOp, RemOp);

// Macro to implement binary operations for numeric types
macro_rules! impl_binary_op {
    ($op:ty, $scalar_op:tt, $simd_method:ident, $elem:ty) => {
        impl SimdBinaryOp<$elem> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(simd: S, a: <$elem as SimdElement>::Simd<S>, b: <$elem as SimdElement>::Simd<S>) -> <$elem as SimdElement>::Simd<S> {
                simd.$simd_method(a, b)
            }

            #[inline(always)]
            fn apply_scalar(a: $elem, b: $elem) -> $elem {
                a $scalar_op b
            }
        }
    };
}

// Implement AddOp for all types
impl_binary_op!(AddOp, +, add_f32s, f32);
impl_binary_op!(AddOp, +, add_f64s, f64);
impl_binary_op!(AddOp, +, add_i8s, i8);
impl_binary_op!(AddOp, +, add_i16s, i16);
impl_binary_op!(AddOp, +, add_i32s, i32);
impl_binary_op!(AddOp, +, add_i64s, i64);
impl_binary_op!(AddOp, +, add_u8s, u8);
impl_binary_op!(AddOp, +, add_u16s, u16);
impl_binary_op!(AddOp, +, add_u32s, u32);
impl_binary_op!(AddOp, +, add_u64s, u64);

// Implement SubOp for all types
impl_binary_op!(SubOp, -, sub_f32s, f32);
impl_binary_op!(SubOp, -, sub_f64s, f64);
impl_binary_op!(SubOp, -, sub_i8s, i8);
impl_binary_op!(SubOp, -, sub_i16s, i16);
impl_binary_op!(SubOp, -, sub_i32s, i32);
impl_binary_op!(SubOp, -, sub_i64s, i64);
impl_binary_op!(SubOp, -, sub_u8s, u8);
impl_binary_op!(SubOp, -, sub_u16s, u16);
impl_binary_op!(SubOp, -, sub_u32s, u32);
impl_binary_op!(SubOp, -, sub_u64s, u64);

// Implement MulOp for types with SIMD multiply
impl_binary_op!(MulOp, *, mul_f32s, f32);
impl_binary_op!(MulOp, *, mul_f64s, f64);
impl_binary_op!(MulOp, *, mul_i16s, i16);
impl_binary_op!(MulOp, *, mul_i32s, i32);
impl_binary_op!(MulOp, *, mul_u16s, u16);
impl_binary_op!(MulOp, *, mul_u32s, u32);

// Implement DivOp for float types only
impl_binary_op!(DivOp, /, div_f32s, f32);
impl_binary_op!(DivOp, /, div_f64s, f64);

// Implement RemOp for integer types (no SIMD instruction, use scalar fallback)
macro_rules! impl_rem_op_scalar {
    ($elem:ty) => {
        impl SimdBinaryOp<$elem> for RemOp {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                _simd: S,
                a: <$elem as SimdElement>::Simd<S>,
                b: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                // Modulo doesn't have SIMD support, so we need to do element-wise
                // This is a fallback that will be slower but correct
                let mut result = a;
                let a_slice = unsafe {
                    std::slice::from_raw_parts(
                        &a as *const _ as *const $elem,
                        std::mem::size_of_val(&a) / std::mem::size_of::<$elem>(),
                    )
                };
                let b_slice = unsafe {
                    std::slice::from_raw_parts(
                        &b as *const _ as *const $elem,
                        std::mem::size_of_val(&b) / std::mem::size_of::<$elem>(),
                    )
                };
                let result_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        &mut result as *mut _ as *mut $elem,
                        std::mem::size_of_val(&result) / std::mem::size_of::<$elem>(),
                    )
                };
                for i in 0..result_slice.len() {
                    result_slice[i] = a_slice[i] % b_slice[i];
                }
                result
            }

            #[inline(always)]
            fn apply_scalar(a: $elem, b: $elem) -> $elem {
                a % b
            }
        }
    };
}

impl_rem_op_scalar!(u32);
impl_rem_op_scalar!(u64);
impl_rem_op_scalar!(i32);
impl_rem_op_scalar!(i64);

// f16 binary ops: f16 has no native pulp SIMD, so `f16::Simd<S> = F16Scalar`
// (single-lane wrapper). Each op forwards through f32 for math correctness.
macro_rules! impl_f16_binary_op {
    ($op:ty, $f:expr) => {
        impl SimdBinaryOp<half::f16> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                _simd: S,
                a: crate::F16Scalar,
                b: crate::F16Scalar,
            ) -> crate::F16Scalar {
                let f: fn(half::f16, half::f16) -> half::f16 = $f;
                crate::F16Scalar(f(a.0, b.0))
            }

            #[inline(always)]
            fn apply_scalar(a: half::f16, b: half::f16) -> half::f16 {
                let f: fn(half::f16, half::f16) -> half::f16 = $f;
                f(a, b)
            }
        }
    };
}

impl_f16_binary_op!(AddOp, |a: half::f16, b: half::f16| a + b);
impl_f16_binary_op!(SubOp, |a: half::f16, b: half::f16| a - b);
impl_f16_binary_op!(MulOp, |a: half::f16, b: half::f16| a * b);
impl_f16_binary_op!(DivOp, |a: half::f16, b: half::f16| half::f16::from_f32(
    a.to_f32() / b.to_f32()
));
impl_f16_binary_op!(RemOp, |a: half::f16, b: half::f16| half::f16::from_f32(
    a.to_f32() % b.to_f32()
));

// Binary tensor operations
define_tensor_op!(@binary Add, AddOp, std_trait = StdAdd);
define_tensor_op!(@binary Sub, SubOp, std_trait = StdSub);
define_tensor_op!(@binary Mul, MulOp, std_trait = StdMul);
define_tensor_op!(@binary Div, DivOp, std_trait = StdDiv);
define_tensor_op!(@binary Rem, RemOp, std_trait = StdRem);
