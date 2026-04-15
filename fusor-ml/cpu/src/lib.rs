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

// Re-export SlidingWindow from fusor-types
pub use fusor_types::SlidingWindow;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_add_operator() {
        // Use Tensor::from_slice directly - cleaner API
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);

        // Use + operator and eval() to get result
        let result = (a + b).to_concrete();

        assert_eq!(result.get([0]), 11.0);
        assert_eq!(result.get([1]), 22.0);
        assert_eq!(result.get([2]), 33.0);
        assert_eq!(result.get([3]), 44.0);
    }

    #[test]
    fn test_tensor_sub_operator() {
        let a: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

        let result = (a - b).to_concrete();

        assert_eq!(result.get([0]), 9.0);
        assert_eq!(result.get([1]), 18.0);
        assert_eq!(result.get([2]), 27.0);
        assert_eq!(result.get([3]), 36.0);
    }

    #[test]
    fn test_tensor_mul_operator() {
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[2.0, 3.0, 4.0, 5.0]);

        let result = (a * b).to_concrete();

        assert_eq!(result.get([0]), 2.0);
        assert_eq!(result.get([1]), 6.0);
        assert_eq!(result.get([2]), 12.0);
        assert_eq!(result.get([3]), 20.0);
    }

    #[test]
    fn test_tensor_div_operator() {
        let a: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[2.0, 4.0, 5.0, 8.0]);

        let result = (a / b).to_concrete();

        assert_eq!(result.get([0]), 5.0);
        assert_eq!(result.get([1]), 5.0);
        assert_eq!(result.get([2]), 6.0);
        assert_eq!(result.get([3]), 5.0);
    }

    #[test]
    fn test_tensor_neg_operator() {
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, -2.0, 3.0, -4.0]);

        let result = (-a).to_concrete();

        assert_eq!(result.get([0]), -1.0);
        assert_eq!(result.get([1]), 2.0);
        assert_eq!(result.get([2]), -3.0);
        assert_eq!(result.get([3]), 4.0);
    }

    #[test]
    fn test_tensor_chained_operators() {
        // Test (a + b) * c with lazy evaluation
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 1.0, 1.0, 1.0]);
        let c: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[2.0, 2.0, 2.0, 2.0]);

        let result = ((a + b) * c).to_concrete();

        // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8, (4+1)*2=10
        assert_eq!(result.get([0]), 4.0);
        assert_eq!(result.get([1]), 6.0);
        assert_eq!(result.get([2]), 8.0);
        assert_eq!(result.get([3]), 10.0);
    }

    #[test]
    fn test_tensor_2d_operators() {
        let a: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

        let result = (a + b).to_concrete();

        assert_eq!(result.get([0, 0]), 11.0);
        assert_eq!(result.get([0, 2]), 33.0);
        assert_eq!(result.get([1, 0]), 44.0);
        assert_eq!(result.get([1, 2]), 66.0);
    }

    #[test]
    fn test_tensor_methods() {
        // Test the new methods on Tensor
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([4], &[10.0, 20.0, 30.0, 40.0]);

        // Test &Tensor + &Tensor operator
        let result = (&a + &b).to_concrete();
        assert_eq!(result.get([0]), 11.0);

        // Test sum reduction
        let c: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.as_ref().sum(), 10.0);

        // Test max/min
        assert_eq!(c.as_ref().max(), 4.0);
        assert_eq!(c.min(), 1.0);
    }

    #[test]
    fn test_tensor_zeros() {
        let t: Tensor<2, ConcreteTensor<f32, 2>> = Tensor::zeros([3, 4]);
        assert_eq!(t.get([0, 0]), 0.0);
        assert_eq!(t.get([2, 3]), 0.0);
    }

    #[test]
    fn test_slice_2d() {
        // Create a 3x4 tensor
        let t: Tensor<2, ConcreteTensor<f32, 2>> = Tensor::from_slice(
            [3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );

        // Slice to get a 2x2 sub-tensor starting at [1, 1]
        let sliced = t.slice([1..3, 1..3]);
        assert_eq!(sliced.get([0, 0]), 6.0);
        assert_eq!(sliced.get([0, 1]), 7.0);
        assert_eq!(sliced.get([1, 0]), 10.0);
        assert_eq!(sliced.get([1, 1]), 11.0);
    }

    #[test]
    fn test_permute_2d() {
        // Create a 2x3 tensor
        let t: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Permute to 3x2
        let permuted = t.permute([1, 0]);
        assert_eq!(permuted.inner().layout().shape(), &[3, 2]);
        assert_eq!(permuted.get([0, 0]), 1.0);
        assert_eq!(permuted.get([0, 1]), 4.0);
        assert_eq!(permuted.get([1, 0]), 2.0);
        assert_eq!(permuted.get([2, 1]), 6.0);
    }

    #[test]
    fn test_transpose_2d() {
        let t: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let transposed = t.t();
        assert_eq!(transposed.inner().layout().shape(), &[3, 2]);
        assert_eq!(transposed.get([0, 0]), 1.0);
        assert_eq!(transposed.get([0, 1]), 4.0);
        assert_eq!(transposed.get([2, 0]), 3.0);
        assert_eq!(transposed.get([2, 1]), 6.0);
    }

    #[test]
    fn test_broadcast_as() {
        // Create a 1x3 tensor
        let t: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([3], &[1.0, 2.0, 3.0]);

        // Broadcast to 2x3
        let broadcasted = t.broadcast_as([2, 3]);
        assert_eq!(broadcasted.inner().layout().shape(), &[2, 3]);
        assert_eq!(broadcasted.get([0, 0]), 1.0);
        assert_eq!(broadcasted.get([0, 2]), 3.0);
        assert_eq!(broadcasted.get([1, 0]), 1.0);
        assert_eq!(broadcasted.get([1, 2]), 3.0);
    }

    #[test]
    fn test_reshape() {
        let t: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Reshape to 2x3
        let reshaped = t.reshape([2, 3]);
        assert_eq!(reshaped.inner().layout().shape(), &[2, 3]);
        assert_eq!(reshaped.get([0, 0]), 1.0);
        assert_eq!(reshaped.get([0, 2]), 3.0);
        assert_eq!(reshaped.get([1, 0]), 4.0);
        assert_eq!(reshaped.get([1, 2]), 6.0);
    }

    #[test]
    fn test_flatten_all() {
        let t: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let flattened = t.flatten_all();
        assert_eq!(flattened.inner().layout().shape(), &[6]);
        assert_eq!(flattened.get([0]), 1.0);
        assert_eq!(flattened.get([5]), 6.0);
    }

    #[test]
    fn test_narrow() {
        let t: Tensor<2, ConcreteTensor<f32, 2>> = Tensor::from_slice(
            [3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );

        // Narrow dimension 0, starting at 1, length 2
        let narrowed = t.narrow(0, 1, 2);
        assert_eq!(narrowed.inner().layout().shape(), &[2, 4]);
        assert_eq!(narrowed.get([0, 0]), 5.0);
        assert_eq!(narrowed.get([1, 3]), 12.0);
    }

    #[test]
    fn test_chunk() {
        let t: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let chunks = t.chunk(3, 0);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].get([0]), 1.0);
        assert_eq!(chunks[0].get([1]), 2.0);
        assert_eq!(chunks[1].get([0]), 3.0);
        assert_eq!(chunks[2].get([1]), 6.0);
    }

    #[test]
    fn test_repeat() {
        let t: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);

        let repeated = t.repeat([2, 3]);
        assert_eq!(repeated.inner().layout().shape(), &[4, 6]);
        assert_eq!(repeated.get([0, 0]), 1.0);
        assert_eq!(repeated.get([0, 2]), 1.0); // Repeated column
        assert_eq!(repeated.get([2, 0]), 1.0); // Repeated row
        assert_eq!(repeated.get([3, 5]), 4.0); // Last element repeated
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        // Start with a 2x1x3 tensor
        let t: Tensor<3, ConcreteTensor<f32, 3>> =
            Tensor::from_slice([2, 1, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Squeeze dimension 1 to get 2x3
        let squeezed: Tensor<2, MapLayout<ConcreteTensor<f32, 3>, 2>> = t.squeeze(1);
        assert_eq!(squeezed.inner().layout().shape(), &[2, 3]);
        assert_eq!(squeezed.get([0, 0]), 1.0);
        assert_eq!(squeezed.get([1, 2]), 6.0);

        // Unsqueeze back to get 2x1x3
        let unsqueezed: Tensor<3, MapLayout<MapLayout<ConcreteTensor<f32, 3>, 2>, 3>> =
            squeezed.unsqueeze(1);
        assert_eq!(unsqueezed.inner().layout().shape(), &[2, 1, 3]);
        assert_eq!(unsqueezed.get([0, 0, 0]), 1.0);
        assert_eq!(unsqueezed.get([1, 0, 2]), 6.0);
    }

    #[test]
    fn test_cat() {
        let a: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        // Cat along dim 0
        let cat_dim0 = Tensor::cat([a.to_concrete(), b.to_concrete()], 0);
        assert_eq!(cat_dim0.inner().layout().shape(), &[4, 3]);
        assert_eq!(cat_dim0.get([0, 0]), 1.0);
        assert_eq!(cat_dim0.get([2, 0]), 7.0);

        let a2: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b2: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        // Cat along dim 1
        let cat_dim1 = Tensor::cat([a2.to_concrete(), b2.to_concrete()], 1);
        assert_eq!(cat_dim1.inner().layout().shape(), &[2, 6]);
        assert_eq!(cat_dim1.get([0, 0]), 1.0);
        assert_eq!(cat_dim1.get([0, 3]), 7.0);
    }

    #[test]
    fn test_stack() {
        let a: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([3], &[1.0, 2.0, 3.0]);
        let b: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([3], &[4.0, 5.0, 6.0]);

        // Stack along dim 0 to get 2x3
        let stacked: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::stack([a.to_concrete(), b.to_concrete()], 0);
        assert_eq!(stacked.inner().layout().shape(), &[2, 3]);
        assert_eq!(stacked.get([0, 0]), 1.0);
        assert_eq!(stacked.get([1, 0]), 4.0);
    }

    #[test]
    fn test_arange() {
        let t: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::arange(0.0f32, 5.0);
        assert_eq!(t.inner().layout().shape(), &[5]);
        assert_eq!(t.get([0]), 0.0);
        assert_eq!(t.get([4]), 4.0);
    }

    #[test]
    fn test_arange_step() {
        let t: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::arange_step(0.0f32, 2.0, 0.5);
        assert_eq!(t.inner().layout().shape(), &[4]);
        assert_eq!(t.get([0]), 0.0);
        assert_eq!(t.get([1]), 0.5);
        assert_eq!(t.get([2]), 1.0);
        assert_eq!(t.get([3]), 1.5);
    }

    #[test]
    fn test_slice_then_arithmetic() {
        // Test SIMD on non-contiguous sliced tensor
        let a: Tensor<2, ConcreteTensor<f32, 2>> = Tensor::from_slice(
            [4, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );

        // Slice to get a 2x2 sub-tensor
        let sliced = a.slice([1..3, 1..3]);

        // Add scalar to sliced tensor
        let result = sliced.add_scalar(10.0).to_concrete();
        assert_eq!(result.get([0, 0]), 16.0); // 6 + 10
        assert_eq!(result.get([0, 1]), 17.0); // 7 + 10
        assert_eq!(result.get([1, 0]), 20.0); // 10 + 10
        assert_eq!(result.get([1, 1]), 21.0); // 11 + 10
    }

    #[test]
    fn test_transpose_then_reshape() {
        // This should force a copy because transpose makes it non-contiguous
        let t: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Transpose to 3x2
        let transposed = t.t();

        // Reshape to 6
        let reshaped = transposed.reshape([6]);
        assert_eq!(reshaped.inner().layout().shape(), &[6]);
        // After transpose, data should be: [1, 4, 2, 5, 3, 6]
        assert_eq!(reshaped.get([0]), 1.0);
        assert_eq!(reshaped.get([1]), 4.0);
        assert_eq!(reshaped.get([2]), 2.0);
    }

    #[test]
    fn test_make_contiguous() {
        let t: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Slice to make non-contiguous
        let sliced = t.slice([0..2, 1..3]);

        // Make contiguous
        let contiguous = sliced.make_contiguous();

        // Verify data is correct
        assert_eq!(contiguous.get([0, 0]), 2.0);
        assert_eq!(contiguous.get([0, 1]), 3.0);
        assert_eq!(contiguous.get([1, 0]), 5.0);
        assert_eq!(contiguous.get([1, 1]), 6.0);
    }

    #[test]
    fn test_expand() {
        // Test expand (alias for broadcast_as)
        let t: Tensor<1, ConcreteTensor<f32, 1>> = Tensor::from_slice([3], &[1.0, 2.0, 3.0]);

        let expanded = t.expand([2, 3]);
        assert_eq!(expanded.inner().layout().shape(), &[2, 3]);
        assert_eq!(expanded.get([0, 0]), 1.0);
        assert_eq!(expanded.get([1, 2]), 3.0);
    }

    #[test]
    fn test_flatten_last_n() {
        // Flatten last 2 dimensions of a 3D tensor
        let t: Tensor<3, ConcreteTensor<f32, 3>> =
            Tensor::from_slice([2, 3, 4], &(0..24).map(|i| i as f32).collect::<Vec<_>>());

        let flattened = t.flatten_last_n::<2, 2>();
        assert_eq!(flattened.inner().layout().shape(), &[2, 12]);
        assert_eq!(flattened.get([0, 0]), 0.0);
        assert_eq!(flattened.get([0, 11]), 11.0);
        assert_eq!(flattened.get([1, 0]), 12.0);
    }

    #[test]
    fn test_flatten_first_n() {
        // Flatten first 2 dimensions of a 3D tensor
        let t: Tensor<3, ConcreteTensor<f32, 3>> =
            Tensor::from_slice([2, 3, 4], &(0..24).map(|i| i as f32).collect::<Vec<_>>());

        let flattened = t.flatten_first_n::<1, 2>();
        assert_eq!(flattened.inner().layout().shape(), &[6, 4]);
        assert_eq!(flattened.get([0, 0]), 0.0);
        assert_eq!(flattened.get([0, 3]), 3.0);
        assert_eq!(flattened.get([5, 3]), 23.0);
    }

    #[test]
    fn test_transpose_then_flatten_attention_pattern() {
        // This tests the exact pattern used in attention output:
        // wv_raw.transpose(1, 2).flatten_last_n::<1, _>()
        // wv_raw shape: [batch, n_head, n_ctx, head_dim]
        // After transpose(1,2): [batch, n_ctx, n_head, head_dim]
        // After flatten_last_n: [batch, n_ctx, n_state] where n_state = n_head * head_dim

        // Create a [1, 2, 3, 4] tensor representing [batch=1, n_head=2, n_ctx=3, head_dim=4]
        let t: Tensor<4, ConcreteTensor<f32, 4>> =
            Tensor::from_slice([1, 2, 3, 4], &(0..24).map(|i| i as f32).collect::<Vec<_>>());

        // Original layout: t[0, head, ctx, dim] at position head*12 + ctx*4 + dim
        // t[0,0,0,0] = 0, t[0,0,0,1] = 1, ..., t[0,0,2,3] = 11
        // t[0,1,0,0] = 12, t[0,1,0,1] = 13, ..., t[0,1,2,3] = 23

        // Transpose(1,2) swaps n_head and n_ctx dimensions
        // Result shape: [1, 3, 2, 4]
        // transposed[0, ctx, head, dim] = original[0, head, ctx, dim]
        let transposed = t.transpose(1, 2);
        assert_eq!(transposed.inner().layout().shape(), &[1, 3, 2, 4]);

        // transposed[0,0,0,0] should be t[0,0,0,0] = 0
        assert_eq!(transposed.get([0, 0, 0, 0]), 0.0);
        // transposed[0,0,1,0] should be t[0,1,0,0] = 12
        assert_eq!(transposed.get([0, 0, 1, 0]), 12.0);
        // transposed[0,1,0,0] should be t[0,0,1,0] = 4
        assert_eq!(transposed.get([0, 1, 0, 0]), 4.0);
        // transposed[0,2,1,3] should be t[0,1,2,3] = 23
        assert_eq!(transposed.get([0, 2, 1, 3]), 23.0);

        // Flatten last 2 dimensions (n_head and head_dim) into n_state
        // Result shape: [1, 3, 8]
        let flattened = transposed.flatten_last_n::<2, 3>();
        assert_eq!(flattened.inner().layout().shape(), &[1, 3, 8]);

        // flattened[0, ctx, state] where state = head*4 + dim
        // flattened[0,0,0] = transposed[0,0,0,0] = 0
        assert_eq!(flattened.get([0, 0, 0]), 0.0);
        // flattened[0,0,4] = transposed[0,0,1,0] = 12
        assert_eq!(flattened.get([0, 0, 4]), 12.0);
        // flattened[0,1,0] = transposed[0,1,0,0] = 4
        assert_eq!(flattened.get([0, 1, 0]), 4.0);
        // flattened[0,2,7] = transposed[0,2,1,3] = 23
        assert_eq!(flattened.get([0, 2, 7]), 23.0);
    }

    #[test]
    fn test_squeeze_dims() {
        // Squeeze two dimensions at once
        let t: Tensor<4, ConcreteTensor<f32, 4>> =
            Tensor::from_slice([1, 3, 1, 4], &(0..12).map(|i| i as f32).collect::<Vec<_>>());

        let squeezed = t.squeeze_dims::<2, 2>([0, 2]);
        assert_eq!(squeezed.inner().layout().shape(), &[3, 4]);
        assert_eq!(squeezed.get([0, 0]), 0.0);
        assert_eq!(squeezed.get([2, 3]), 11.0);
    }

    #[test]
    fn test_unsqueeze_dims() {
        // Unsqueeze two dimensions at once
        let t: Tensor<2, ConcreteTensor<f32, 2>> =
            Tensor::from_slice([3, 4], &(0..12).map(|i| i as f32).collect::<Vec<_>>());

        let unsqueezed = t.unsqueeze_dims::<2, 4>([0, 2]);
        assert_eq!(unsqueezed.inner().layout().shape(), &[1, 3, 1, 4]);
        assert_eq!(unsqueezed.get([0, 0, 0, 0]), 0.0);
        assert_eq!(unsqueezed.get([0, 2, 0, 3]), 11.0);
    }

    #[test]
    fn test_sliding_window_view_1d() {
        // Test 1D sliding window
        let t: Tensor<1, ConcreteTensor<f32, 1>> =
            Tensor::from_slice([7], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        let windows = t.sliding_window_view::<1, 2>([SlidingWindow::new(0, 3, 2)]);

        // (7 - 3) / 2 + 1 = 3 positions, window size 3
        assert_eq!(windows.inner().layout().shape(), &[3, 3]);

        // First window: [1, 2, 3]
        assert_eq!(windows.get([0, 0]), 1.0);
        assert_eq!(windows.get([0, 1]), 2.0);
        assert_eq!(windows.get([0, 2]), 3.0);

        // Second window: [3, 4, 5]
        assert_eq!(windows.get([1, 0]), 3.0);
        assert_eq!(windows.get([1, 1]), 4.0);
        assert_eq!(windows.get([1, 2]), 5.0);

        // Third window: [5, 6, 7]
        assert_eq!(windows.get([2, 0]), 5.0);
        assert_eq!(windows.get([2, 1]), 6.0);
        assert_eq!(windows.get([2, 2]), 7.0);
    }

    #[test]
    fn test_sliding_window_view_2d() {
        // Test 2D sliding window
        let data: Vec<f32> = (1..=36).map(|i| i as f32).collect();
        let t: Tensor<2, ConcreteTensor<f32, 2>> = Tensor::from_slice([6, 6], &data);

        let windows = t.sliding_window_view::<2, 4>([
            SlidingWindow::new(0, 3, 3),
            SlidingWindow::new(1, 3, 3),
        ]);

        // (6 - 3) / 3 + 1 = 2 positions in each dimension
        assert_eq!(windows.inner().layout().shape(), &[2, 2, 3, 3]);

        // Verify some values
        assert_eq!(windows.get([0, 0, 0, 0]), 1.0); // Top-left of first window
        assert_eq!(windows.get([0, 0, 2, 2]), 15.0); // Bottom-right of first 3x3 window
        assert_eq!(windows.get([0, 1, 0, 0]), 4.0); // Top-left of second window (row 0, col 1)
        assert_eq!(windows.get([1, 0, 0, 0]), 19.0); // Top-left of window at (1, 0)
    }
}
