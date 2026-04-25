//! Elementwise (unary) tensor operations: Neg, Abs, Sqrt

use std::ops::Neg as StdNeg;

use pulp::Simd;

use crate::{ConcreteTensor, SimdElement, TensorBacking, materialize_expr};
use fusor_types::Layout;

/// Trait for unary operations that have SIMD support
pub trait SimdUnaryOp<E: SimdElement>: Copy {
    /// Apply operation to SIMD vector
    fn apply_simd_vec<S: Simd>(simd: S, a: E::Simd<S>) -> E::Simd<S>;

    /// Apply operation to scalar
    fn apply_scalar(val: E) -> E;
}

// Unary operation markers
macro_rules! define_op_marker {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Copy, Clone)]
            pub struct $name;
        )*
    };
}
define_op_marker!(
    NegOp, AbsOp, SqrtOp, ExpOp, Exp2Op, LogOp, Log2Op, SinOp, CosOp, TanOp, TanhOp, AsinOp,
    AcosOp, AtanOp, SinhOp, CoshOp, AsinhOp, AcoshOp, AtanhOp
);

// Macro for unary ops with SIMD support
macro_rules! impl_unary_op {
    ($op:ty, $scalar_fn:expr, $simd_method:ident, $elem:ty) => {
        impl SimdUnaryOp<$elem> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                simd: S,
                a: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                simd.$simd_method(a)
            }

            #[inline(always)]
            fn apply_scalar(val: $elem) -> $elem {
                let f: fn($elem) -> $elem = $scalar_fn;
                f(val)
            }
        }
    };
}

// NegOp implementations
impl_unary_op!(NegOp, |x: f32| -x, neg_f32s, f32);
impl_unary_op!(NegOp, |x: f64| -x, neg_f64s, f64);

// NegOp for integer types using subtraction from zero
macro_rules! impl_neg_int_op {
    ($elem:ty, $splat:ident, $sub:ident) => {
        impl SimdUnaryOp<$elem> for NegOp {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                simd: S,
                a: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                simd.$sub(simd.$splat(0), a)
            }

            #[inline(always)]
            fn apply_scalar(val: $elem) -> $elem {
                val.wrapping_neg()
            }
        }
    };
}

impl_neg_int_op!(i8, splat_i8s, sub_i8s);
impl_neg_int_op!(i16, splat_i16s, sub_i16s);
impl_neg_int_op!(i32, splat_i32s, sub_i32s);
impl_neg_int_op!(i64, splat_i64s, sub_i64s);

// AbsOp for floats (native SIMD support)
impl_unary_op!(AbsOp, |x: f32| x.abs(), abs_f32s, f32);
impl_unary_op!(AbsOp, |x: f64| x.abs(), abs_f64s, f64);

// AbsOp for integers using max(x, -x)
macro_rules! impl_abs_int_op {
    ($elem:ty, $splat:ident, $sub:ident, $max:ident) => {
        impl SimdUnaryOp<$elem> for AbsOp {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                simd: S,
                a: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                let zero = simd.$splat(0);
                let neg_a = simd.$sub(zero, a);
                simd.$max(a, neg_a)
            }

            #[inline(always)]
            fn apply_scalar(val: $elem) -> $elem {
                val.wrapping_abs()
            }
        }
    };
}

impl_abs_int_op!(i8, splat_i8s, sub_i8s, max_i8s);
impl_abs_int_op!(i16, splat_i16s, sub_i16s, max_i16s);
impl_abs_int_op!(i32, splat_i32s, sub_i32s, max_i32s);
impl_abs_int_op!(i64, splat_i64s, sub_i64s, max_i64s);

// Sqrt for floats
impl_unary_op!(SqrtOp, |x: f32| x.sqrt(), sqrt_f32s, f32);
impl_unary_op!(SqrtOp, |x: f64| x.sqrt(), sqrt_f64s, f64);

// Macro for scalar-only unary ops (no SIMD intrinsic available)
// Uses scalar evaluation per SIMD lane, which still benefits from fusion
macro_rules! impl_scalar_unary_op {
    ($op:ty, $scalar_fn:expr, $elem:ty) => {
        impl SimdUnaryOp<$elem> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(
                _simd: S,
                a: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                // Process each lane with scalar operation
                let lane_count = std::mem::size_of::<<$elem as SimdElement>::Simd<S>>()
                    / std::mem::size_of::<$elem>();
                let mut temp = [<$elem>::default(); crate::MAX_SIMD_LANES];

                // Safe: cast SIMD ref to scalar slice via bytemuck
                let input_slice: &[$elem] = pulp::bytemuck::cast_slice(std::slice::from_ref(&a));
                temp[..lane_count].copy_from_slice(input_slice);

                let f: fn($elem) -> $elem = $scalar_fn;
                for i in 0..lane_count {
                    temp[i] = f(temp[i]);
                }

                // Safe: reconstruct SIMD from scalar slice via as_simd
                let (simd_slice, _) = <$elem as SimdElement>::as_simd::<S>(&temp[..lane_count]);
                simd_slice[0]
            }

            #[inline(always)]
            fn apply_scalar(val: $elem) -> $elem {
                let f: fn($elem) -> $elem = $scalar_fn;
                f(val)
            }
        }
    };
}

// Transcendental ops for f32
impl_scalar_unary_op!(ExpOp, |x: f32| x.exp(), f32);
impl_scalar_unary_op!(Exp2Op, |x: f32| x.exp2(), f32);
impl_scalar_unary_op!(LogOp, |x: f32| x.ln(), f32);
impl_scalar_unary_op!(Log2Op, |x: f32| x.log2(), f32);
impl_scalar_unary_op!(SinOp, |x: f32| x.sin(), f32);
impl_scalar_unary_op!(CosOp, |x: f32| x.cos(), f32);
impl_scalar_unary_op!(TanOp, |x: f32| x.tan(), f32);
impl_scalar_unary_op!(TanhOp, |x: f32| x.tanh(), f32);

// Transcendental ops for f64
impl_scalar_unary_op!(ExpOp, |x: f64| x.exp(), f64);
impl_scalar_unary_op!(Exp2Op, |x: f64| x.exp2(), f64);
impl_scalar_unary_op!(LogOp, |x: f64| x.ln(), f64);
impl_scalar_unary_op!(Log2Op, |x: f64| x.log2(), f64);
impl_scalar_unary_op!(SinOp, |x: f64| x.sin(), f64);
impl_scalar_unary_op!(CosOp, |x: f64| x.cos(), f64);
impl_scalar_unary_op!(TanOp, |x: f64| x.tan(), f64);
impl_scalar_unary_op!(TanhOp, |x: f64| x.tanh(), f64);

// Additional inverse trig and hyperbolic ops for f32
impl_scalar_unary_op!(AsinOp, |x: f32| x.asin(), f32);
impl_scalar_unary_op!(AcosOp, |x: f32| x.acos(), f32);
impl_scalar_unary_op!(AtanOp, |x: f32| x.atan(), f32);
impl_scalar_unary_op!(SinhOp, |x: f32| x.sinh(), f32);
impl_scalar_unary_op!(CoshOp, |x: f32| x.cosh(), f32);
impl_scalar_unary_op!(AsinhOp, |x: f32| x.asinh(), f32);
impl_scalar_unary_op!(AcoshOp, |x: f32| x.acosh(), f32);
impl_scalar_unary_op!(AtanhOp, |x: f32| x.atanh(), f32);

// Additional inverse trig and hyperbolic ops for f64
impl_scalar_unary_op!(AsinOp, |x: f64| x.asin(), f64);
impl_scalar_unary_op!(AcosOp, |x: f64| x.acos(), f64);
impl_scalar_unary_op!(AtanOp, |x: f64| x.atan(), f64);
impl_scalar_unary_op!(SinhOp, |x: f64| x.sinh(), f64);
impl_scalar_unary_op!(CoshOp, |x: f64| x.cosh(), f64);
impl_scalar_unary_op!(AsinhOp, |x: f64| x.asinh(), f64);
impl_scalar_unary_op!(AcoshOp, |x: f64| x.acosh(), f64);
impl_scalar_unary_op!(AtanhOp, |x: f64| x.atanh(), f64);

// f16 unary ops: pulp has no native f16 SIMD, so `f16::Simd<S> = F16Scalar`
// (a single-lane wrapper). Each op forwards through f32 for math correctness
// and re-rounds to f16. See cpu/src/lib.rs:F16Scalar.
macro_rules! impl_f16_unary_op {
    ($op:ty, $f:expr) => {
        impl SimdUnaryOp<half::f16> for $op {
            #[inline(always)]
            fn apply_simd_vec<S: Simd>(_simd: S, a: crate::F16Scalar) -> crate::F16Scalar {
                let f: fn(half::f16) -> half::f16 = $f;
                crate::F16Scalar(f(a.0))
            }

            #[inline(always)]
            fn apply_scalar(val: half::f16) -> half::f16 {
                let f: fn(half::f16) -> half::f16 = $f;
                f(val)
            }
        }
    };
}

impl_f16_unary_op!(NegOp, |x: half::f16| -x);
impl_f16_unary_op!(AbsOp, |x: half::f16| half::f16::from_f32(x.to_f32().abs()));
impl_f16_unary_op!(SqrtOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().sqrt()
));
impl_f16_unary_op!(ExpOp, |x: half::f16| half::f16::from_f32(x.to_f32().exp()));
impl_f16_unary_op!(Exp2Op, |x: half::f16| half::f16::from_f32(
    x.to_f32().exp2()
));
impl_f16_unary_op!(LogOp, |x: half::f16| half::f16::from_f32(x.to_f32().ln()));
impl_f16_unary_op!(Log2Op, |x: half::f16| half::f16::from_f32(
    x.to_f32().log2()
));
impl_f16_unary_op!(SinOp, |x: half::f16| half::f16::from_f32(x.to_f32().sin()));
impl_f16_unary_op!(CosOp, |x: half::f16| half::f16::from_f32(x.to_f32().cos()));
impl_f16_unary_op!(TanOp, |x: half::f16| half::f16::from_f32(x.to_f32().tan()));
impl_f16_unary_op!(TanhOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().tanh()
));
impl_f16_unary_op!(AsinOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().asin()
));
impl_f16_unary_op!(AcosOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().acos()
));
impl_f16_unary_op!(AtanOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().atan()
));
impl_f16_unary_op!(SinhOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().sinh()
));
impl_f16_unary_op!(CoshOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().cosh()
));
impl_f16_unary_op!(AsinhOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().asinh()
));
impl_f16_unary_op!(AcoshOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().acosh()
));
impl_f16_unary_op!(AtanhOp, |x: half::f16| half::f16::from_f32(
    x.to_f32().atanh()
));

/// Macro to define unary tensor operations (Neg, Abs, Sqrt)
macro_rules! define_unary_tensor_op {
    ($name:ident, $simd_op:ty) => {
        pub struct $name<E: SimdElement, const R: usize, T: TensorBacking<R, Elem = E>> {
            input: T,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const R: usize, T> $name<E, R, T>
        where
            E: SimdElement,
            T: TensorBacking<R, Elem = E>,
        {
            pub fn new(input: T) -> Self {
                Self {
                    input,
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<E, const R: usize, T> crate::LazyBacking for $name<E, R, T>
        where
            E: SimdElement + Default,
            $simd_op: SimdUnaryOp<E>,
            T: TensorBacking<R, Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op>::apply_scalar(self.input.eval_scalar(idx))
            }

            #[inline(always)]
            fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op>::apply_simd_vec(simd, self.input.eval_simd(simd, base_idx))
            }
        }

        impl<E, const R: usize, T> TensorBacking<R> for $name<E, R, T>
        where
            E: SimdElement + Default,
            $simd_op: SimdUnaryOp<E>,
            T: TensorBacking<R, Elem = E>,
        {
            fn layout(&self) -> Layout {
                Layout::contiguous(self.input.layout().shape())
            }

            fn to_concrete(&self) -> ConcreteTensor<E, R> {
                let shape: [usize; R] = self
                    .input
                    .layout()
                    .shape()
                    .try_into()
                    .expect("Shape length mismatch");
                materialize_expr(self, shape)
            }
        }
    };
    ($name:ident, $simd_op:ty, $std_trait:ident) => {
        pub struct $name<E: SimdElement, const R: usize, T: TensorBacking<R, Elem = E>> {
            input: T,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const R: usize, T> $name<E, R, T>
        where
            E: SimdElement,
            T: TensorBacking<R, Elem = E>,
        {
            pub fn new(input: T) -> Self {
                Self {
                    input,
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<E, const R: usize, T> crate::LazyBacking for $name<E, R, T>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdUnaryOp<E>,
            T: TensorBacking<R, Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op>::apply_scalar(self.input.eval_scalar(idx))
            }

            #[inline(always)]
            fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op>::apply_simd_vec(simd, self.input.eval_simd(simd, base_idx))
            }
        }

        impl<E, const R: usize, T> TensorBacking<R> for $name<E, R, T>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            $simd_op: SimdUnaryOp<E>,
            T: TensorBacking<R, Elem = E>,
        {
            fn layout(&self) -> Layout {
                Layout::contiguous(self.input.layout().shape())
            }

            fn to_concrete(&self) -> ConcreteTensor<E, R> {
                let shape: [usize; R] = self
                    .input
                    .layout()
                    .shape()
                    .try_into()
                    .expect("Shape length mismatch");
                materialize_expr(self, shape)
            }
        }
    };
}

// Unary tensor operations
define_unary_tensor_op!(Neg, NegOp, StdNeg);
define_unary_tensor_op!(Abs, AbsOp);
define_unary_tensor_op!(Sqrt, SqrtOp);

// Transcendental tensor operations
define_unary_tensor_op!(Exp, ExpOp);
define_unary_tensor_op!(Exp2, Exp2Op);
define_unary_tensor_op!(Log, LogOp);
define_unary_tensor_op!(Log2, Log2Op);
define_unary_tensor_op!(Sin, SinOp);
define_unary_tensor_op!(Cos, CosOp);
define_unary_tensor_op!(Tan, TanOp);
define_unary_tensor_op!(Tanh, TanhOp);

// Additional inverse trig and hyperbolic tensor operations
define_unary_tensor_op!(Asin, AsinOp);
define_unary_tensor_op!(Acos, AcosOp);
define_unary_tensor_op!(Atan, AtanOp);
define_unary_tensor_op!(Sinh, SinhOp);
define_unary_tensor_op!(Cosh, CoshOp);
define_unary_tensor_op!(Asinh, AsinhOp);
define_unary_tensor_op!(Acosh, AcoshOp);
define_unary_tensor_op!(Atanh, AtanhOp);
