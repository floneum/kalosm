//! ConcreteTensor - the actual tensor storage implementation

use aligned_vec::{ABox, AVec};
use fusor_types::Layout;
use pulp::Simd;

use crate::expr::linear_to_indices;
use crate::{ResolvedTensor, SimdElement, TensorBacking};

/// Helper to iterate over indices of a tensor with given shape
pub struct IndexIterator {
    shape: Box<[usize]>,
    indices: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    pub fn new(shape: &[usize]) -> Self {
        let done = shape.contains(&0);
        Self {
            shape: shape.into(),
            indices: vec![0; shape.len()],
            done,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.indices.clone();

        // Special case: 0D tensor (scalar) - yield once then done
        if self.shape.is_empty() {
            self.done = true;
            return Some(result);
        }

        // Increment indices (row-major order)
        for i in (0..self.shape.len()).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.shape[i] {
                break;
            }
            self.indices[i] = 0;
            if i == 0 {
                self.done = true;
            }
        }

        Some(result)
    }
}

#[derive(Clone)]
pub struct ConcreteTensor<T, const R: usize> {
    layout: Layout,
    backing: ABox<[T]>,
}

impl<T, const R: usize> crate::LazyBacking for ConcreteTensor<T, R>
where
    T: SimdElement,
{
    type Elem = T;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> T {
        if self.layout.is_contiguous() {
            self.backing[idx]
        } else {
            // Convert linear index to logical indices for strided access
            let shape: &[usize; R] = self.layout.shape().try_into().unwrap();
            let indices = linear_to_indices::<R>(idx, shape);
            let phys_idx = self.layout.linear_index(&indices);
            self.backing[phys_idx]
        }
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> T::Simd<S> {
        if self.layout.is_contiguous() {
            // SAFETY: Caller guarantees base_idx is aligned to SIMD width and within bounds.
            // The backing is allocated with 64-byte alignment via aligned-vec.
            // ConcreteTensor is always contiguous - non-contiguous access uses MapLayout.
            unsafe { *self.backing.as_ptr().add(base_idx).cast::<T::Simd<S>>() }
        } else {
            // Optimized path: use SIMD gather for strided tensor access
            // Precompute physical indices for all SIMD lanes
            let lane_count = const { std::mem::size_of::<T::Simd<S>>() / std::mem::size_of::<T>() };
            let shape: &[usize; R] = self.layout.shape().try_into().unwrap();
            let phys_indices: [usize; crate::MAX_SIMD_LANES] = std::array::from_fn(|i| {
                let indices = linear_to_indices::<R>(base_idx + i, shape);
                self.layout.linear_index(&indices)
            });

            // Use SIMD gather instruction to load all elements at once
            // SAFETY: All indices are computed from valid linear indices
            // within the tensor's logical bounds, and the backing array
            // contains all physical positions that the layout can address.
            unsafe { T::gather_unchecked(simd, &self.backing, &phys_indices, lane_count) }
        }
    }
}

impl<T, const R: usize> TensorBacking<R> for ConcreteTensor<T, R>
where
    T: SimdElement,
{
    fn layout(&self) -> Layout {
        self.layout.clone()
    }

    fn to_concrete(&self) -> ConcreteTensor<T, R> {
        self.clone()
    }
}

impl<T, const R: usize> ResolvedTensor<R> for ConcreteTensor<T, R>
where
    T: SimdElement,
{
    fn data(&self) -> &ABox<[Self::Elem]> {
        &self.backing
    }
    fn data_mut(&mut self) -> &mut ABox<[Self::Elem]> {
        &mut self.backing
    }
}

impl<T: SimdElement, const R: usize> ConcreteTensor<T, R> {
    /// Create a new tensor with contiguous layout from shape, filled with zeros
    pub fn zeros(shape: [usize; R]) -> Self
    where
        T: Default,
    {
        let layout = Layout::contiguous(&shape);
        let num_elements = layout.num_elements();
        let mut vec: AVec<T> = AVec::with_capacity(64, num_elements);
        vec.resize(num_elements, T::default());
        let backing = vec.into_boxed_slice();
        Self { layout, backing }
    }

    /// Create a new tensor from existing data with contiguous layout
    pub fn from_slice(shape: [usize; R], data: &[T]) -> Self {
        let layout = Layout::contiguous(&shape);
        assert_eq!(layout.num_elements(), data.len());
        let mut vec: AVec<T> = AVec::with_capacity(64, data.len());
        vec.extend_from_slice(data);
        let backing = vec.into_boxed_slice();
        Self { layout, backing }
    }

    /// Get a reference to the layout
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Calculate the linear index for given logical indices
    fn linear_index(&self, indices: &[usize; R]) -> usize {
        self.layout.linear_index(indices)
    }

    /// Get element at logical indices
    pub fn get(&self, indices: [usize; R]) -> T {
        let idx = self.linear_index(&indices);
        self.backing[idx]
    }

    /// Set element at logical indices
    pub fn set(&mut self, indices: [usize; R], value: T) {
        let idx = self.linear_index(&indices);
        self.backing[idx] = value;
    }

    /// Create a new tensor from existing layout and backing data.
    pub fn from_parts(layout: Layout, backing: ABox<[T]>) -> Self {
        Self { layout, backing }
    }

    /// Get the backing data
    pub(crate) fn backing(&self) -> &ABox<[T]> {
        &self.backing
    }

    /// Get the backing data mutably
    pub(crate) fn backing_mut(&mut self) -> &mut ABox<[T]> {
        &mut self.backing
    }

    /// Create a new tensor with contiguous layout from shape, filled by a closure.
    ///
    /// The closure receives the linear index and returns the element value.
    pub(crate) fn from_fn(shape: [usize; R], mut f: impl FnMut(usize) -> T) -> Self {
        let layout = Layout::contiguous(&shape);
        let n = layout.num_elements();
        let mut vec: AVec<T> = AVec::with_capacity(64, n);
        for i in 0..n {
            vec.push(f(i));
        }
        Self {
            layout,
            backing: vec.into_boxed_slice(),
        }
    }
}
