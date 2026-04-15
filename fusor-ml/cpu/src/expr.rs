//! Expression template system for lazy operation fusion
//!
//! This module provides types that enable automatic fusion of elementwise
//! operations. When multiple operations are chained (e.g.,
//! `x.mul_ref(&y).add_ref(&z).sqrt_ref()`), they are evaluated in a single
//! SIMD loop instead of multiple passes.

use pulp::{Arch, Simd, WithSimd};

use crate::{ConcreteTensor, SimdElement, TensorBacking};

/// Helper to get SIMD lane count for a given element type and SIMD architecture
#[inline(always)]
fn simd_lane_count<E: SimdElement, S: Simd>() -> usize {
    std::mem::size_of::<E::Simd<S>>() / std::mem::size_of::<E>()
}

/// Evaluates a tensor backing into output memory using SIMD.
///
/// This is the core evaluation loop that fuses all operations in an
/// expression tree into a single pass over the data.
struct TensorEvaluator<'a, T: TensorBacking<R>, const R: usize> {
    tensor: &'a T,
    out: &'a mut [T::Elem],
    /// Logical offset into the tensor for this chunk
    base_offset: usize,
}

impl<T: TensorBacking<R>, const R: usize> WithSimd for TensorEvaluator<'_, T, R> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (simd_out, scalar_out) = T::Elem::as_mut_simd::<S>(self.out);

        // Main SIMD loop
        let lane_count = simd_lane_count::<T::Elem, S>();
        for (i, out_vec) in simd_out.iter_mut().enumerate() {
            let base_idx = self.base_offset + i * lane_count;
            *out_vec = self.tensor.eval_simd(simd, base_idx);
        }

        // Scalar tail - handles remaining elements
        let scalar_start = simd_out.len() * lane_count;
        for (i, out_val) in scalar_out.iter_mut().enumerate() {
            *out_val = self.tensor.eval_scalar(self.base_offset + scalar_start + i);
        }
    }
}

/// Minimum number of elements before parallelization is used.
/// Below this threshold, the overhead of thread spawning isn't worth it.
/// Note: Set very high because std::thread::scope has significant overhead
/// (spawns new threads each time). For repeated operations like in transformer
/// layers, the thread spawn/join cost dominates. A thread pool would be better
/// but for now we avoid parallelization for typical LLM tensor sizes.
const PARALLEL_THRESHOLD: usize = 16_777_216; // 16M elements (~64MB for f32)

/// Materialize a tensor backing into a new ConcreteTensor.
///
/// This evaluates the entire expression tree in a single fused SIMD loop,
/// writing the results into a newly allocated tensor.
///
/// For large tensors (>4096 elements), the work is split among multiple
/// threads using `std::thread::scope` for structured parallelism.
#[inline]
#[must_use = "this allocates a new tensor; discarding it wastes computation"]
pub fn materialize_expr<T: TensorBacking<R> + Sync, const R: usize>(
    tensor: &T,
    shape: [usize; R],
) -> ConcreteTensor<T::Elem, R> {
    let mut output = ConcreteTensor::<T::Elem, R>::zeros(shape);
    let total_elements = output.backing().len();

    let n_threads = crate::parallel::num_threads();

    // Use parallel execution for large tensors
    if total_elements >= PARALLEL_THRESHOLD && n_threads > 1 {
        // Align chunk boundaries to MAX_SIMD_LANES to maintain SIMD alignment across chunks
        let raw_chunk = total_elements.div_ceil(n_threads);
        let chunk_size = raw_chunk.next_multiple_of(crate::MAX_SIMD_LANES);
        let out_slice: &mut [T::Elem] = output.backing_mut();

        std::thread::scope(|scope| {
            let mut remaining_slice = out_slice;
            let mut offset = 0;

            for thread_id in 0..n_threads {
                if remaining_slice.is_empty() {
                    break;
                }

                let remaining = total_elements - offset;
                let this_size = if thread_id == n_threads - 1 {
                    remaining
                } else {
                    chunk_size.min(remaining)
                };

                let (thread_slice, rest) = remaining_slice.split_at_mut(this_size);
                remaining_slice = rest;
                let current_offset = offset;
                offset += this_size;

                let evaluator = TensorEvaluator {
                    tensor,
                    out: thread_slice,
                    base_offset: current_offset,
                };

                scope.spawn(move || {
                    Arch::new().dispatch(evaluator);
                });
            }
        });
    } else {
        // Small tensor: single-threaded execution
        let out_slice: &mut [T::Elem] = output.backing_mut();
        Arch::new().dispatch(TensorEvaluator {
            tensor,
            out: out_slice,
            base_offset: 0,
        });
    }

    output
}

/// Convert a linear index to logical indices for a given shape.
///
/// This is used for strided tensor access where we need to map
/// a flat iteration index to multi-dimensional tensor coordinates.
#[inline]
pub(crate) fn linear_to_indices<const R: usize>(
    mut linear: usize,
    shape: &[usize; R],
) -> [usize; R] {
    let mut indices = [0usize; R];

    // Work backwards through dimensions (row-major order)
    for i in (0..R).rev() {
        indices[i] = linear % shape[i];
        linear /= shape[i];
    }

    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LazyBacking;

    #[test]
    fn test_linear_to_indices_1d() {
        let shape = [5];
        assert_eq!(linear_to_indices::<1>(0, &shape), [0]);
        assert_eq!(linear_to_indices::<1>(4, &shape), [4]);
    }

    #[test]
    fn test_linear_to_indices_2d() {
        let shape = [3, 4]; // 3 rows, 4 cols
        assert_eq!(linear_to_indices::<2>(0, &shape), [0, 0]);
        assert_eq!(linear_to_indices::<2>(1, &shape), [0, 1]);
        assert_eq!(linear_to_indices::<2>(4, &shape), [1, 0]);
        assert_eq!(linear_to_indices::<2>(11, &shape), [2, 3]);
    }

    #[test]
    fn test_linear_to_indices_3d() {
        let shape = [2, 3, 4]; // 2x3x4 tensor
        assert_eq!(linear_to_indices::<3>(0, &shape), [0, 0, 0]);
        assert_eq!(linear_to_indices::<3>(1, &shape), [0, 0, 1]);
        assert_eq!(linear_to_indices::<3>(4, &shape), [0, 1, 0]);
        assert_eq!(linear_to_indices::<3>(12, &shape), [1, 0, 0]);
    }

    #[test]
    fn test_concrete_tensor_eval() {
        let tensor = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

        // Test layout methods
        assert_eq!(tensor.layout().num_elements(), 4);
        assert_eq!(tensor.layout().shape(), &[4]);
        assert!(tensor.layout().is_contiguous());

        // Test scalar evaluation
        assert_eq!(tensor.eval_scalar(0), 1.0);
        assert_eq!(tensor.eval_scalar(1), 2.0);
        assert_eq!(tensor.eval_scalar(3), 4.0);
    }

    #[test]
    fn test_materialize_expr_basic() {
        let tensor = ConcreteTensor::<f32, 1>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]);

        // Materialize a simple expression (identity)
        let result: ConcreteTensor<f32, 1> = materialize_expr(&tensor, [4]);

        for i in 0..4 {
            assert_eq!(result.get([i]), tensor.get([i]));
        }
    }

    #[test]
    fn test_reference_impl() {
        let tensor = ConcreteTensor::<f32, 1>::from_slice([3], &[1.0, 2.0, 3.0]);
        let tensor_ref = &tensor;

        // Test that references implement TensorBacking correctly
        assert_eq!(tensor_ref.layout().num_elements(), 3);
        assert_eq!(tensor_ref.layout().shape(), &[3]);
        assert!(tensor_ref.layout().is_contiguous());
        assert_eq!(tensor_ref.eval_scalar(0), 1.0);
        assert_eq!(tensor_ref.eval_scalar(2), 3.0);
    }
}
