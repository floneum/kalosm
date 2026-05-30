//! CPU tensor operations with SIMD acceleration

use std::ops::Deref;

use pulp::Simd;
use pulp::bytemuck::Pod;

// Module declarations
mod backing;
mod cast;
mod comparison;
mod concrete_tensor;
mod conditional;
mod dynamic;
mod elementwise;
mod expr;
mod gather;
mod index;
mod map_layout;
mod matmul;
mod pairwise;
mod parallel;
mod quantized;
mod rank;
mod reduce;
mod scalar;
mod slice_assign;
mod tensor;

/// Maximum number of SIMD lanes supported for strided tensor gather operations.
/// This covers AVX-512 with 64 x i8 lanes. Current architectures don't exceed this,
/// but this constant provides a clear point for future updates if needed.
pub(crate) const MAX_SIMD_LANES: usize = 64;

// Public dynamic tensor API.
pub use dynamic::{
    CpuDType, CpuElement, DynamicTensor, DynamicTensor as Tensor, DynamicTensorError,
};
pub use fusor_types::Layout;

// Internal typed fusion/storage API used by this crate and the fusor facade.
#[allow(unused_imports)]
pub(crate) use backing::{LazyBacking, ResolvedTensor, TensorBacking};
#[allow(unused_imports)]
pub(crate) use concrete_tensor::ConcreteTensor;
#[allow(unused_imports)]
pub(crate) use elementwise::{
    Abs, Acos, Acosh, Asin, Asinh, Atan, Atanh, Cos, Cosh, Exp, Exp2, Log, Log2, Neg, Sin, Sinh,
    Sqrt, Tan, Tanh,
};
#[allow(unused_imports)]
pub(crate) use expr::materialize_expr;
#[allow(unused_imports)]
pub(crate) use map_layout::MapLayout;
#[allow(unused_imports)]
pub(crate) use pairwise::{Add, Div, Mul, Rem, Sub};
#[allow(unused_imports)]
pub(crate) use quantized::{Dequantize, QuantizedTensor};
#[allow(unused_imports)]
pub(crate) use rank::{
    LargerRank, LargerRankInner, LastRank, LastRankInner, MaxRank, MaxRankInner, NextRank,
    NextRankInner, SmallerRank, SmallerRankInner,
};
#[allow(unused_imports)]
pub(crate) use scalar::{AddScalar, Broadcast, DivScalar, MulScalar, SubScalar};
#[allow(unused_imports)]
pub(crate) use tensor::{FloatOps, Scalar, TypedTensor};

// Re-export FromArray trait from fusor-types
pub use fusor_types::FromArray;

// Re-export aligned_vec types for use by dependent crates
pub use aligned_vec::ABox;
pub use aligned_vec::AVec;

// Re-export GGUF types for convenience
pub use fusor_gguf::{
    BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType, GgufBlock,
};

// Re-export TensorSlice from fusor-types
pub use fusor_types::TensorSlice;

/// A buffer holding CPU tensor data as bytes.
///
/// This type is the CPU equivalent of fusor-core's `MappedBuffer` for GPU tensors.
/// It holds the raw bytes of tensor data and implements `Deref<Target = [u8]>`
/// to work with `TensorSlice`.
pub struct CpuMappedBuffer {
    bytes: Box<[u8]>,
}

impl CpuMappedBuffer {
    /// Create a new CpuMappedBuffer from a boxed byte slice.
    pub fn new(bytes: Box<[u8]>) -> Self {
        Self { bytes }
    }
}

impl Deref for CpuMappedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.bytes
    }
}

// Internal operation traits and markers.
#[allow(unused_imports)]
pub(crate) use cast::CastTo;
#[allow(unused_imports)]
pub(crate) use comparison::{Eq, EqOp, Gt, GtOp, Gte, GteOp, Lt, LtOp, Lte, LteOp, Ne, NeOp};
#[allow(unused_imports)]
pub(crate) use conditional::IsNonZero;
#[allow(unused_imports)]
pub(crate) use elementwise::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CosOp, CoshOp, Exp2Op, ExpOp, Log2Op,
    LogOp, NegOp, SimdUnaryOp, SinOp, SinhOp, SqrtOp, TanOp, TanhOp,
};
#[allow(unused_imports)]
pub(crate) use matmul::MatmulImpl;
#[allow(unused_imports)]
pub(crate) use pairwise::{AddOp, DivOp, MulOp, RemOp, SimdBinaryOp, SubOp};
#[allow(unused_imports)]
pub(crate) use reduce::{
    MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp, layer_norm_last_dim_fused, softmax_last_dim_fused,
};

// Re-export internal types used by other modules
pub(crate) use concrete_tensor::IndexIterator;

#[doc(hidden)]
pub mod __private {
    pub use crate::backing::{LazyBacking, ResolvedTensor, TensorBacking};
    pub use crate::cast::CastTo;
    pub use crate::comparison::{Eq, EqOp, Gt, GtOp, Gte, GteOp, Lt, LtOp, Lte, LteOp, Ne, NeOp};
    pub use crate::concrete_tensor::ConcreteTensor;
    pub use crate::conditional::IsNonZero;
    pub use crate::elementwise::{
        Abs, AbsOp, Acos, AcosOp, Acosh, AcoshOp, Asin, AsinOp, Asinh, AsinhOp, Atan, AtanOp,
        Atanh, AtanhOp, Cos, CosOp, Cosh, CoshOp, Exp, Exp2, Exp2Op, ExpOp, Log, Log2, Log2Op,
        LogOp, Neg, NegOp, SimdUnaryOp, Sin, SinOp, Sinh, SinhOp, Sqrt, SqrtOp, Tan, TanOp, Tanh,
        TanhOp,
    };
    pub use crate::expr::materialize_expr;
    pub use crate::map_layout::MapLayout;
    pub use crate::matmul::MatmulImpl;
    pub use crate::pairwise::{
        Add, AddOp, Div, DivOp, Mul, MulOp, Rem, RemOp, SimdBinaryOp, Sub, SubOp,
    };
    pub use crate::quantized::{Dequantize, QuantizedTensor};
    pub use crate::rank::{
        LargerRank, LargerRankInner, LastRank, LastRankInner, MaxRank, MaxRankInner, NextRank,
        NextRankInner, SmallerRank, SmallerRankInner,
    };
    pub use crate::reduce::{
        MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp, layer_norm_last_dim_fused,
        softmax_last_dim_fused,
    };
    pub use crate::scalar::{AddScalar, Broadcast, DivScalar, MulScalar, SubScalar};
    pub use crate::tensor::{FloatOps, Scalar, TypedTensor};
    pub use crate::{F16Scalar, SimdElement};
}

/// Canonical macro for defining lazy tensor expression types.
///
/// Subsumes the four arities of tensor-op expressions:
/// - `@unary`: one tensor input (e.g. `Neg`, `Abs`, `Sqrt`, transcendentals)
/// - `@binary`: two tensor inputs (e.g. `Add`, `Sub`, comparisons)
/// - `@scalar`: one tensor input + one scalar (e.g. `AddScalar`)
///
/// Each variant accepts an optional `, std_trait = $StdTrait` fragment that
/// adds `E: $StdTrait<Output = E>` to the trait-impl where-clauses. Pairwise
/// ops use this to require e.g. `E: StdAdd<Output = E>`; comparisons and most
/// elementwise ops omit it.
///
/// The `LazyBacking` impls keep `#[inline(always)]` on both `eval_scalar` and
/// `eval_simd` so kernel fusion still inlines through the wrapper structs.
macro_rules! define_tensor_op {
    // -------- Unary (single tensor input) --------
    (
        @unary $name:ident, $simd_op:ty $(, std_trait = $std_trait:ident)?
    ) => {
        pub struct $name<E: $crate::SimdElement, const R: usize, T: $crate::TensorBacking<R, Elem = E>> {
            input: T,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const R: usize, T> $name<E, R, T>
        where
            E: $crate::SimdElement,
            T: $crate::TensorBacking<R, Elem = E>,
        {
            pub fn new(input: T) -> Self {
                Self {
                    input,
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<E, const R: usize, T> $crate::LazyBacking for $name<E, R, T>
        where
            E: $crate::SimdElement + Default $(+ $std_trait<Output = E>)?,
            $simd_op: $crate::elementwise::SimdUnaryOp<E>,
            T: $crate::TensorBacking<R, Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op as $crate::elementwise::SimdUnaryOp<E>>::apply_scalar(
                    self.input.eval_scalar(idx),
                )
            }

            #[inline(always)]
            fn eval_simd<S: pulp::Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op as $crate::elementwise::SimdUnaryOp<E>>::apply_simd_vec(
                    simd,
                    self.input.eval_simd(simd, base_idx),
                )
            }
        }

        impl<E, const R: usize, T> $crate::TensorBacking<R> for $name<E, R, T>
        where
            E: $crate::SimdElement + Default $(+ $std_trait<Output = E>)?,
            $simd_op: $crate::elementwise::SimdUnaryOp<E>,
            T: $crate::TensorBacking<R, Elem = E>,
        {
            fn layout(&self) -> $crate::Layout {
                $crate::Layout::contiguous(self.input.layout().shape())
            }

            fn to_concrete(&self) -> $crate::ConcreteTensor<E, R> {
                let shape: [usize; R] = self
                    .input
                    .layout()
                    .shape()
                    .try_into()
                    .expect("Shape length mismatch");
                $crate::materialize_expr(self, shape)
            }
        }
    };

    // -------- Binary (two tensor inputs) --------
    (
        @binary $name:ident, $simd_op:ty $(, std_trait = $std_trait:ident)?
    ) => {
        pub struct $name<
            E: $crate::SimdElement,
            const R: usize,
            T1: $crate::TensorBacking<R, Elem = E>,
            T2: $crate::TensorBacking<R, Elem = E>,
        > {
            lhs: T1,
            rhs: T2,
            _marker: std::marker::PhantomData<E>,
        }

        impl<E, const R: usize, T1, T2> $name<E, R, T1, T2>
        where
            E: $crate::SimdElement,
            T1: $crate::TensorBacking<R, Elem = E>,
            T2: $crate::TensorBacking<R, Elem = E>,
        {
            pub fn new(lhs: T1, rhs: T2) -> Self {
                Self {
                    lhs,
                    rhs,
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<E, const R: usize, T1, T2> $crate::LazyBacking for $name<E, R, T1, T2>
        where
            E: $crate::SimdElement + Default $(+ $std_trait<Output = E>)?,
            $simd_op: $crate::pairwise::SimdBinaryOp<E>,
            T1: $crate::TensorBacking<R, Elem = E>,
            T2: $crate::TensorBacking<R, Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op as $crate::pairwise::SimdBinaryOp<E>>::apply_scalar(
                    self.lhs.eval_scalar(idx),
                    self.rhs.eval_scalar(idx),
                )
            }

            #[inline(always)]
            fn eval_simd<S: pulp::Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op as $crate::pairwise::SimdBinaryOp<E>>::apply_simd_vec(
                    simd,
                    self.lhs.eval_simd(simd, base_idx),
                    self.rhs.eval_simd(simd, base_idx),
                )
            }
        }

        impl<E, const R: usize, T1, T2> $crate::TensorBacking<R> for $name<E, R, T1, T2>
        where
            E: $crate::SimdElement + Default $(+ $std_trait<Output = E>)?,
            $simd_op: $crate::pairwise::SimdBinaryOp<E>,
            T1: $crate::TensorBacking<R, Elem = E>,
            T2: $crate::TensorBacking<R, Elem = E>,
        {
            fn layout(&self) -> $crate::Layout {
                $crate::Layout::contiguous(self.lhs.layout().shape())
            }

            fn to_concrete(&self) -> $crate::ConcreteTensor<E, R> {
                let shape: [usize; R] = self
                    .lhs
                    .layout()
                    .shape()
                    .try_into()
                    .expect("Shape length mismatch");
                $crate::materialize_expr(self, shape)
            }
        }
    };

    // -------- Scalar (tensor + scalar value) --------
    (
        @scalar $name:ident, $simd_op:ty $(, std_trait = $std_trait:ident)?
    ) => {
        pub struct $name<E: $crate::SimdElement, const R: usize, T: $crate::TensorBacking<R, Elem = E>> {
            tensor: T,
            scalar: E,
        }

        impl<E, const R: usize, T> $name<E, R, T>
        where
            E: $crate::SimdElement,
            T: $crate::TensorBacking<R, Elem = E>,
        {
            pub fn new(tensor: T, scalar: E) -> Self {
                Self { tensor, scalar }
            }
        }

        impl<E, const R: usize, T> $crate::LazyBacking for $name<E, R, T>
        where
            E: $crate::SimdElement + Default $(+ $std_trait<Output = E>)?,
            $simd_op: $crate::pairwise::SimdBinaryOp<E>,
            T: $crate::TensorBacking<R, Elem = E>,
        {
            type Elem = E;

            #[inline(always)]
            fn eval_scalar(&self, idx: usize) -> E {
                <$simd_op as $crate::pairwise::SimdBinaryOp<E>>::apply_scalar(
                    self.tensor.eval_scalar(idx),
                    self.scalar,
                )
            }

            #[inline(always)]
            fn eval_simd<S: pulp::Simd>(&self, simd: S, base_idx: usize) -> E::Simd<S> {
                <$simd_op as $crate::pairwise::SimdBinaryOp<E>>::apply_simd_vec(
                    simd,
                    self.tensor.eval_simd(simd, base_idx),
                    <E as $crate::SimdElement>::splat(simd, self.scalar),
                )
            }
        }

        impl<E, const R: usize, T> $crate::TensorBacking<R> for $name<E, R, T>
        where
            E: $crate::SimdElement + Default $(+ $std_trait<Output = E>)?,
            $simd_op: $crate::pairwise::SimdBinaryOp<E>,
            T: $crate::TensorBacking<R, Elem = E>,
        {
            fn layout(&self) -> $crate::Layout {
                $crate::Layout::contiguous(self.tensor.layout().shape())
            }

            fn to_concrete(&self) -> $crate::ConcreteTensor<E, R> {
                let shape: [usize; R] = self
                    .tensor
                    .layout()
                    .shape()
                    .try_into()
                    .expect("Shape length mismatch");
                $crate::materialize_expr(self, shape)
            }
        }
    };
}

pub(crate) use define_tensor_op;

/// Trait for SIMD element types with associated SIMD vector type
pub trait SimdElement: Sized + Copy + Default + Pod + Sync + Send {
    /// The SIMD vector type for this element (GAT)
    type Simd<S: Simd>: Copy;

    /// Convert slice to SIMD vectors + remainder
    fn as_simd<S: Simd>(slice: &[Self]) -> (&[Self::Simd<S>], &[Self]);
    fn as_mut_simd<S: Simd>(slice: &mut [Self]) -> (&mut [Self::Simd<S>], &mut [Self]);

    /// Broadcast a scalar value to all lanes of a SIMD vector
    fn splat<S: Simd>(simd: S, value: Self) -> Self::Simd<S>;

    /// Gather elements from the slice at the specified indices using SIMD.
    ///
    /// # Safety
    /// All indices must be valid indices into the slice.
    ///
    /// # Arguments
    /// * `simd` - The SIMD context
    /// * `slice` - The source data slice
    /// * `indices` - Array of indices to gather from
    /// * `lane_count` - Number of SIMD lanes to fill
    ///
    /// Uses hardware SIMD gather instructions (AVX2, AVX-512) when available,
    /// falling back to scalar loads on other architectures.
    unsafe fn gather_unchecked<S: Simd>(
        simd: S,
        slice: &[Self],
        indices: &[usize],
        lane_count: usize,
    ) -> Self::Simd<S>;
}

macro_rules! impl_simd_element {
    ($elem:ty, $simd_ty:ident, $as_simd:ident, $as_mut_simd:ident, $splat:ident) => {
        impl SimdElement for $elem {
            type Simd<S: Simd> = S::$simd_ty;

            #[inline(always)]
            fn as_simd<S: Simd>(slice: &[Self]) -> (&[S::$simd_ty], &[Self]) {
                S::$as_simd(slice)
            }

            #[inline(always)]
            fn as_mut_simd<S: Simd>(slice: &mut [Self]) -> (&mut [S::$simd_ty], &mut [Self]) {
                S::$as_mut_simd(slice)
            }

            #[inline(always)]
            fn splat<S: Simd>(simd: S, value: Self) -> S::$simd_ty {
                simd.$splat(value)
            }

            #[inline(always)]
            unsafe fn gather_unchecked<S: Simd>(
                simd: S,
                slice: &[Self],
                indices: &[usize],
                lane_count: usize,
            ) -> Self::Simd<S> {
                // SAFETY: Caller guarantees all indices are valid
                unsafe { gather::gather_impl::<Self, S>(simd, slice, indices, lane_count) }
            }
        }
    };
}

impl_simd_element!(f32, f32s, as_simd_f32s, as_mut_simd_f32s, splat_f32s);
impl_simd_element!(f64, f64s, as_simd_f64s, as_mut_simd_f64s, splat_f64s);
impl_simd_element!(i8, i8s, as_simd_i8s, as_mut_simd_i8s, splat_i8s);
impl_simd_element!(i16, i16s, as_simd_i16s, as_mut_simd_i16s, splat_i16s);
impl_simd_element!(i32, i32s, as_simd_i32s, as_mut_simd_i32s, splat_i32s);
impl_simd_element!(i64, i64s, as_simd_i64s, as_mut_simd_i64s, splat_i64s);
impl_simd_element!(u8, u8s, as_simd_u8s, as_mut_simd_u8s, splat_u8s);
impl_simd_element!(u16, u16s, as_simd_u16s, as_mut_simd_u16s, splat_u16s);
impl_simd_element!(u32, u32s, as_simd_u32s, as_mut_simd_u32s, splat_u32s);
impl_simd_element!(u64, u64s, as_simd_u64s, as_mut_simd_u64s, splat_u64s);

/// Wrapper type for f16 "SIMD" operations.
///
/// Since pulp doesn't have native f16 SIMD support, we use a scalar fallback.
/// This type wraps a single f16 value and presents itself as a "SIMD vector"
/// with one lane. Operations fall back to scalar code.
#[derive(Copy, Clone)]
pub struct F16Scalar(pub half::f16);

impl SimdElement for half::f16 {
    /// The "SIMD" type for f16 is just a scalar wrapper since pulp lacks f16 SIMD.
    type Simd<S: Simd> = F16Scalar;

    #[inline(always)]
    fn as_simd<S: Simd>(_slice: &[Self]) -> (&[Self::Simd<S>], &[Self]) {
        // No SIMD for f16, return empty SIMD slice and all elements as remainder
        (&[], _slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(_slice: &mut [Self]) -> (&mut [Self::Simd<S>], &mut [Self]) {
        // No SIMD for f16, return empty SIMD slice and all elements as remainder
        (&mut [], _slice)
    }

    #[inline(always)]
    fn splat<S: Simd>(_simd: S, value: Self) -> Self::Simd<S> {
        F16Scalar(value)
    }

    #[inline(always)]
    unsafe fn gather_unchecked<S: Simd>(
        _simd: S,
        slice: &[Self],
        indices: &[usize],
        _lane_count: usize,
    ) -> Self::Simd<S> {
        // Scalar fallback: just return the first element
        F16Scalar(slice[indices[0]])
    }
}
