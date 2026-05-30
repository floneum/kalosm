use aligned_vec::ABox;
use fusor_types::Layout;
use pulp::Simd;

use crate::{ConcreteTensor, SimdElement};

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
