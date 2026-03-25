//! CPU tensor operations with SIMD acceleration

use std::ops::Deref;

use pulp::Simd;
use pulp::bytemuck::Pod;

// Module declarations
mod cast;
mod comparison;
mod concrete_tensor;
mod conditional;
mod elementwise;
mod expr;
mod gather;
mod index;
mod map_layout;
mod matmul;
mod pairwise;
mod parallel;
mod quantized;
mod reduce;
mod scalar;
mod slice_assign;
mod tensor;

/// Maximum number of SIMD lanes supported for strided tensor gather operations.
/// This covers AVX-512 with 64 x i8 lanes. Current architectures don't exceed this,
/// but this constant provides a clear point for future updates if needed.
pub(crate) const MAX_SIMD_LANES: usize = 64;

// Re-export public types
pub use concrete_tensor::ConcreteTensor;
pub use elementwise::{
    Abs, Acos, Acosh, Asin, Asinh, Atan, Atanh, Cos, Cosh, Exp, Exp2, Log, Log2, Neg, Sin, Sinh,
    Sqrt, Tan, Tanh,
};
pub use expr::materialize_expr;
pub use map_layout::MapLayout;
pub use pairwise::{Add, Div, Mul, Rem, Sub};
pub use quantized::{Dequantize, QuantizedTensor};
pub use scalar::{AddScalar, Broadcast, DivScalar, MulScalar, SubScalar};
pub use tensor::{FloatOps, Scalar, Tensor};

// Re-export FromArray trait from fusor-types
pub use fusor_types::FromArray;

// Re-export Layout from fusor-types for public API
pub use fusor_types::Layout;

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

// Re-export operation traits and markers for public bounds
pub use cast::CastTo;
pub use comparison::{Eq, EqOp, Gt, GtOp, Gte, GteOp, Lt, LtOp, Lte, LteOp, Ne, NeOp};
pub use conditional::IsNonZero;
pub use elementwise::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CosOp, CoshOp, Exp2Op, ExpOp, Log2Op,
    LogOp, NegOp, SimdUnaryOp, SinOp, SinhOp, SqrtOp, TanOp, TanhOp,
};
pub use matmul::MatmulImpl;
pub use pairwise::{AddOp, DivOp, MulOp, RemOp, SimdBinaryOp, SubOp};
pub use reduce::{
    MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp, layer_norm_last_dim_fused, softmax_last_dim_fused,
};

// Re-export internal types used by other modules
pub(crate) use concrete_tensor::IndexIterator;

// Trait for mapping tensor to its one-rank-smaller type (for axis reductions)
pub trait LastRankInner {
    type LastRank;
}

pub trait LastRank<const R: usize, T: SimdElement>:
    LastRankInner<LastRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, T: SimdElement, X> LastRank<R, T> for X where
    X: LastRankInner<LastRank = ConcreteTensor<T, R>>
{
}

// Macro to generate LastRankInner implementations for each rank
macro_rules! impl_last_rank {
    ($($R:literal),*) => {
        $(
            impl<T: SimdElement> LastRankInner for ConcreteTensor<T, $R> {
                type LastRank = ConcreteTensor<T, { $R - 1 }>;
            }
        )*
    };
}

// Generate for ranks 1-10
impl_last_rank!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

// Trait for mapping tensor to its next-higher rank type (for unsqueeze)
pub trait NextRankInner {
    type NextRank;
}

pub trait NextRank<const R: usize, T: SimdElement>:
    NextRankInner<NextRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, T: SimdElement, X> NextRank<R, T> for X where
    X: NextRankInner<NextRank = ConcreteTensor<T, R>>
{
}

// Macro to generate NextRankInner implementations for each rank
macro_rules! impl_next_rank {
    ($($R:literal),*) => {
        $(
            impl<T: SimdElement> NextRankInner for ConcreteTensor<T, $R> {
                type NextRank = ConcreteTensor<T, { $R + 1 }>;
            }
        )*
    };
}

// Generate for ranks 0-9 (so next rank goes up to 10)
impl_next_rank!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

// Trait for mapping tensor to a smaller rank (for squeeze, reduce)
pub trait SmallerRankInner<const DIFF: usize> {
    type SmallerRank;
}

pub trait SmallerRank<const R: usize, const DIFF: usize, T: SimdElement>:
    SmallerRankInner<DIFF, SmallerRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, const DIFF: usize, T: SimdElement, X> SmallerRank<R, DIFF, T> for X where
    X: SmallerRankInner<DIFF, SmallerRank = ConcreteTensor<T, R>>
{
}

// Macro to generate SmallerRankInner implementations
macro_rules! impl_smaller_rank {
    ($R:literal, $($DIFF:literal => $OUT:literal),*) => {
        $(
            impl<T: SimdElement> SmallerRankInner<$DIFF> for ConcreteTensor<T, $R> {
                type SmallerRank = ConcreteTensor<T, $OUT>;
            }
        )*
    };
}

// Generate smaller rank mappings
impl_smaller_rank!(1, 1 => 0);
impl_smaller_rank!(2, 1 => 1, 2 => 0);
impl_smaller_rank!(3, 1 => 2, 2 => 1, 3 => 0);
impl_smaller_rank!(4, 1 => 3, 2 => 2, 3 => 1, 4 => 0);
impl_smaller_rank!(5, 1 => 4, 2 => 3, 3 => 2, 4 => 1, 5 => 0);
impl_smaller_rank!(6, 1 => 5, 2 => 4, 3 => 3, 4 => 2, 5 => 1, 6 => 0);
impl_smaller_rank!(7, 1 => 6, 2 => 5, 3 => 4, 4 => 3, 5 => 2, 6 => 1, 7 => 0);
impl_smaller_rank!(8, 1 => 7, 2 => 6, 3 => 5, 4 => 4, 5 => 3, 6 => 2, 7 => 1, 8 => 0);
impl_smaller_rank!(9, 1 => 8, 2 => 7, 3 => 6, 4 => 5, 5 => 4, 6 => 3, 7 => 2, 8 => 1, 9 => 0);
impl_smaller_rank!(10, 1 => 9, 2 => 8, 3 => 7, 4 => 6, 5 => 5, 6 => 4, 7 => 3, 8 => 2, 9 => 1, 10 => 0);

// Trait for mapping tensor to a larger rank (for unsqueeze, expand)
pub trait LargerRankInner<const DIFF: usize> {
    type LargerRank;
}

pub trait LargerRank<const R: usize, const DIFF: usize, T: SimdElement>:
    LargerRankInner<DIFF, LargerRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, const DIFF: usize, T: SimdElement, X> LargerRank<R, DIFF, T> for X where
    X: LargerRankInner<DIFF, LargerRank = ConcreteTensor<T, R>>
{
}

// Macro to generate LargerRankInner implementations
macro_rules! impl_larger_rank {
    ($R:literal, $($DIFF:literal => $OUT:literal),*) => {
        $(
            impl<T: SimdElement> LargerRankInner<$DIFF> for ConcreteTensor<T, $R> {
                type LargerRank = ConcreteTensor<T, $OUT>;
            }
        )*
    };
}

// Generate larger rank mappings
impl_larger_rank!(0, 1 => 1, 2 => 2, 3 => 3, 4 => 4, 5 => 5, 6 => 6, 7 => 7, 8 => 8, 9 => 9, 10 => 10);
impl_larger_rank!(1, 1 => 2, 2 => 3, 3 => 4, 4 => 5, 5 => 6, 6 => 7, 7 => 8, 8 => 9, 9 => 10);
impl_larger_rank!(2, 1 => 3, 2 => 4, 3 => 5, 4 => 6, 5 => 7, 6 => 8, 7 => 9, 8 => 10);
impl_larger_rank!(3, 1 => 4, 2 => 5, 3 => 6, 4 => 7, 5 => 8, 6 => 9, 7 => 10);
impl_larger_rank!(4, 1 => 5, 2 => 6, 3 => 7, 4 => 8, 5 => 9, 6 => 10);
impl_larger_rank!(5, 1 => 6, 2 => 7, 3 => 8, 4 => 9, 5 => 10);
impl_larger_rank!(6, 1 => 7, 2 => 8, 3 => 9, 4 => 10);
impl_larger_rank!(7, 1 => 8, 2 => 9, 3 => 10);
impl_larger_rank!(8, 1 => 9, 2 => 10);
impl_larger_rank!(9, 1 => 10);

// Trait for mapping two tensors to their max rank (for broadcasting operations)
pub trait MaxRankInner {
    type MaxRank;
}

pub trait MaxRank<const R: usize, T: SimdElement>:
    MaxRankInner<MaxRank = ConcreteTensor<T, R>>
{
}

impl<const R: usize, T: SimdElement, X> MaxRank<R, T> for X where
    X: MaxRankInner<MaxRank = ConcreteTensor<T, R>>
{
}

// Same rank produces same rank
impl<const N: usize, T: SimdElement> MaxRankInner for (ConcreteTensor<T, N>, ConcreteTensor<T, N>) {
    type MaxRank = ConcreteTensor<T, N>;
}

// Macro to generate MaxRankInner implementations for different rank pairs
macro_rules! impl_max_rank {
    ($R1:literal, $R2:literal) => {
        impl<T: SimdElement> MaxRankInner for (ConcreteTensor<T, $R1>, ConcreteTensor<T, $R2>) {
            type MaxRank = ConcreteTensor<T, $R2>;
        }
        impl<T: SimdElement> MaxRankInner for (ConcreteTensor<T, $R2>, ConcreteTensor<T, $R1>) {
            type MaxRank = ConcreteTensor<T, $R2>;
        }
    };
}

// Generate MaxRank implementations for all rank combinations 0-10
impl_max_rank!(0, 1);
impl_max_rank!(0, 2);
impl_max_rank!(0, 3);
impl_max_rank!(0, 4);
impl_max_rank!(0, 5);
impl_max_rank!(0, 6);
impl_max_rank!(0, 7);
impl_max_rank!(0, 8);
impl_max_rank!(0, 9);
impl_max_rank!(0, 10);
impl_max_rank!(1, 2);
impl_max_rank!(1, 3);
impl_max_rank!(1, 4);
impl_max_rank!(1, 5);
impl_max_rank!(1, 6);
impl_max_rank!(1, 7);
impl_max_rank!(1, 8);
impl_max_rank!(1, 9);
impl_max_rank!(1, 10);
impl_max_rank!(2, 3);
impl_max_rank!(2, 4);
impl_max_rank!(2, 5);
impl_max_rank!(2, 6);
impl_max_rank!(2, 7);
impl_max_rank!(2, 8);
impl_max_rank!(2, 9);
impl_max_rank!(2, 10);
impl_max_rank!(3, 4);
impl_max_rank!(3, 5);
impl_max_rank!(3, 6);
impl_max_rank!(3, 7);
impl_max_rank!(3, 8);
impl_max_rank!(3, 9);
impl_max_rank!(3, 10);
impl_max_rank!(4, 5);
impl_max_rank!(4, 6);
impl_max_rank!(4, 7);
impl_max_rank!(4, 8);
impl_max_rank!(4, 9);
impl_max_rank!(4, 10);
impl_max_rank!(5, 6);
impl_max_rank!(5, 7);
impl_max_rank!(5, 8);
impl_max_rank!(5, 9);
impl_max_rank!(5, 10);
impl_max_rank!(6, 7);
impl_max_rank!(6, 8);
impl_max_rank!(6, 9);
impl_max_rank!(6, 10);
impl_max_rank!(7, 8);
impl_max_rank!(7, 9);
impl_max_rank!(7, 10);
impl_max_rank!(8, 9);
impl_max_rank!(8, 10);
impl_max_rank!(9, 10);

/// Trait for types that support scalar and SIMD evaluation without a rank parameter.
/// This is a supertrait of `TensorBacking` that allows rank-independent access.
pub trait LazyBacking: Sync {
    type Elem: SimdElement;

    /// Evaluate at a single scalar index.
    ///
    /// This is used for:
    /// - Tail elements that don't fill a complete SIMD vector
    /// - Non-contiguous tensor access patterns
    fn eval_scalar(&self, idx: usize) -> Self::Elem;

    /// Evaluate a SIMD chunk starting at the given base index.
    ///
    /// The returned SIMD vector contains multiple consecutive elements
    /// starting at `base_idx`. The caller must ensure that there are
    /// enough elements remaining to fill a complete SIMD vector.
    fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> <Self::Elem as SimdElement>::Simd<S>;
}

pub trait TensorBacking<const R: usize>: LazyBacking {
    fn layout(&self) -> Layout;
    fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R>;
}

// Blanket implementation for references
impl<T: LazyBacking + Sync> LazyBacking for &T {
    type Elem = T::Elem;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> Self::Elem {
        (*self).eval_scalar(idx)
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> <Self::Elem as SimdElement>::Simd<S> {
        (*self).eval_simd(simd, base_idx)
    }
}

impl<const R: usize, T: TensorBacking<R> + Sync> TensorBacking<R> for &T {
    fn layout(&self) -> Layout {
        (*self).layout()
    }

    fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R> {
        (*self).to_concrete()
    }
}

pub trait ResolvedTensor<const R: usize>: TensorBacking<R> {
    fn data(&self) -> &ABox<[Self::Elem]>;
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]>;
}

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
