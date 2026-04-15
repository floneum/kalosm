//! MapLayout - A lazy tensor wrapper that holds backing data with a transformed layout
//!
//! This module provides `MapLayout`, which enables O(1) layout operations (slice, permute,
//! transpose, broadcast, etc.) by storing the backing data with a new layout instead of
//! cloning the data.

use fusor_types::Layout;
use pulp::Simd;

use crate::expr::{linear_to_indices, materialize_expr};
use crate::{ConcreteTensor, LazyBacking, MAX_SIMD_LANES, SimdElement, TensorBacking};

/// A tensor that holds backing data with a transformed layout.
///
/// `MapLayout` is created by layout-changing operations like `slice`, `permute`,
/// `transpose`, `broadcast_as`, and `reshape`. It wraps any `LazyBacking` and applies a
/// layout transformation, enabling O(1) lazy layout operations.
///
/// This allows chaining layout operations without materializing intermediate results:
/// `tensor.exp().slice([...]).transpose(0, 1).reshape([...])` preserves laziness throughout.
///
/// # Type Parameters
/// - `T`: The backing type (implements `LazyBacking`)
/// - `R`: The output rank of this tensor
pub struct MapLayout<T, const R: usize> {
    backing: T,
    layout: Layout,
}

impl<T: Clone, const R: usize> Clone for MapLayout<T, R> {
    fn clone(&self) -> Self {
        Self {
            backing: self.backing.clone(),
            layout: self.layout.clone(),
        }
    }
}

impl<T, const R: usize> MapLayout<T, R> {
    /// Create a new MapLayout from backing data and a layout.
    pub(crate) fn new(backing: T, layout: Layout) -> Self {
        Self { backing, layout }
    }

    /// Get a reference to the layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: LazyBacking, const R: usize> MapLayout<T, R> {
    /// Get element at logical indices.
    pub fn get(&self, indices: [usize; R]) -> T::Elem {
        let idx = self.layout.linear_index(&indices);
        self.backing.eval_scalar(idx)
    }
}

impl<T: LazyBacking, const R: usize> LazyBacking for MapLayout<T, R> {
    type Elem = T::Elem;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> Self::Elem {
        if self.layout.is_contiguous() {
            self.backing.eval_scalar(idx)
        } else {
            let shape: &[usize; R] = self.layout.shape().try_into().unwrap();
            let logical_indices = linear_to_indices::<R>(idx, shape);
            let physical_idx = self.layout.linear_index(&logical_indices);
            self.backing.eval_scalar(physical_idx)
        }
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> <Self::Elem as SimdElement>::Simd<S> {
        if self.layout.is_contiguous() {
            self.backing.eval_simd(simd, base_idx)
        } else {
            // Gather elements via scalar evaluation for strided access
            let lane_count = std::mem::size_of::<<Self::Elem as SimdElement>::Simd<S>>()
                / std::mem::size_of::<Self::Elem>();

            let shape: &[usize; R] = self.layout.shape().try_into().unwrap();

            let mut temp = [Self::Elem::default(); MAX_SIMD_LANES];
            for (i, temp_elem) in temp.iter_mut().enumerate().take(lane_count) {
                let logical_indices = linear_to_indices::<R>(base_idx + i, shape);
                let physical_idx = self.layout.linear_index(&logical_indices);
                *temp_elem = self.backing.eval_scalar(physical_idx);
            }

            let (simd_slice, _) = Self::Elem::as_simd::<S>(&temp[..lane_count]);
            simd_slice[0]
        }
    }
}

impl<T: LazyBacking, const R: usize> TensorBacking<R> for MapLayout<T, R> {
    fn layout(&self) -> Layout {
        Layout::contiguous(self.layout.shape())
    }

    fn to_concrete(&self) -> ConcreteTensor<Self::Elem, R> {
        let shape: [usize; R] = self.layout.shape().try_into().expect("Shape mismatch");
        materialize_expr(self, shape)
    }
}
