//! Slice assign operation: replace a slice region with values from another tensor

use std::ops::Range;

use crate::{ConcreteTensor, ResolvedTensor, SimdElement};

/// Validate that the slice bounds are within the input tensor shape
/// and that the value tensor shape matches the slice dimensions
fn validate_slice_assign<const R: usize>(
    input_shape: &[usize],
    slices: &[Range<usize>; R],
    value_shape: &[usize],
) {
    assert_eq!(input_shape.len(), R, "Input shape rank mismatch");
    assert_eq!(value_shape.len(), R, "Value shape rank mismatch");

    for i in 0..R {
        // Check slice bounds
        assert!(
            slices[i].start <= slices[i].end,
            "Slice start must be <= end at dimension {}: {}..{}",
            i,
            slices[i].start,
            slices[i].end
        );
        assert!(
            slices[i].end <= input_shape[i],
            "Slice end {} exceeds input dimension {} size {}",
            slices[i].end,
            i,
            input_shape[i]
        );

        // Check value shape matches slice size
        let slice_size = slices[i].end - slices[i].start;
        assert_eq!(
            value_shape[i], slice_size,
            "Value shape mismatch at dimension {}: expected {} (slice size), got {}",
            i, slice_size, value_shape[i]
        );
    }
}

/// Check if a set of indices falls within the slice region
#[inline]
fn is_in_slice<const R: usize>(indices: &[usize; R], slices: &[Range<usize>; R]) -> bool {
    for i in 0..R {
        if indices[i] < slices[i].start || indices[i] >= slices[i].end {
            return false;
        }
    }
    true
}

/// Convert output indices to value tensor indices (subtract slice start)
#[inline]
fn to_value_indices<const R: usize>(
    indices: &[usize; R],
    slices: &[Range<usize>; R],
) -> [usize; R] {
    let mut value_indices = [0usize; R];
    for i in 0..R {
        value_indices[i] = indices[i] - slices[i].start;
    }
    value_indices
}

/// Slice assign: return a new tensor with the slice region replaced by values from the value tensor
///
/// For a 3x3 tensor with value 2x2 at slices [0..2, 0..2]:
/// ```text
/// [[1, 2, 3],      [[10, 11, 3],
///  [4, 5, 6],  =>   [12, 13, 6],
///  [7, 8, 9]]       [7,  8,  9]]
/// ```
pub(crate) fn slice_assign_ref<E, const R: usize>(
    input: &ConcreteTensor<E, R>,
    slices: [Range<usize>; R],
    value: &ConcreteTensor<E, R>,
) -> ConcreteTensor<E, R>
where
    E: SimdElement,
{
    let input_shape = input.layout().shape();
    let value_shape = value.layout().shape();

    validate_slice_assign::<R>(input_shape, &slices, value_shape);

    // Check if both tensors are contiguous for potential fast path
    let input_contiguous = input.layout().is_contiguous();
    let value_contiguous = value.layout().is_contiguous();

    if input_contiguous && value_contiguous {
        slice_assign_contiguous(input, &slices, value)
    } else {
        slice_assign_strided(input, &slices, value)
    }
}

/// Fast path for contiguous tensors
fn slice_assign_contiguous<E, const R: usize>(
    input: &ConcreteTensor<E, R>,
    slices: &[Range<usize>; R],
    value: &ConcreteTensor<E, R>,
) -> ConcreteTensor<E, R>
where
    E: SimdElement,
{
    let input_shape: [usize; R] = input
        .layout()
        .shape()
        .try_into()
        .expect("Shape length mismatch");

    let input_strides = input.layout().strides();
    let value_strides = value.layout().strides();

    ConcreteTensor::from_fn(input_shape, |out_linear| {
        // Convert linear index to multi-dimensional indices
        let mut indices = [0usize; R];
        let mut remaining = out_linear;
        for i in 0..R {
            indices[i] = remaining / input_strides[i];
            remaining %= input_strides[i];
        }

        // Check if this position is in the slice region
        if is_in_slice(&indices, slices) {
            // Get from value tensor
            let value_indices = to_value_indices(&indices, slices);
            let value_linear: usize = value_indices
                .iter()
                .zip(value_strides.iter())
                .map(|(&idx, &stride)| idx * stride)
                .sum();
            value.data()[value_linear]
        } else {
            // Copy from input tensor
            input.data()[out_linear]
        }
    })
}

/// General path for strided tensors
fn slice_assign_strided<E, const R: usize>(
    input: &ConcreteTensor<E, R>,
    slices: &[Range<usize>; R],
    value: &ConcreteTensor<E, R>,
) -> ConcreteTensor<E, R>
where
    E: SimdElement,
{
    let input_shape: [usize; R] = input
        .layout()
        .shape()
        .try_into()
        .expect("Shape length mismatch");

    let output_layout = fusor_types::Layout::contiguous(&input_shape);
    let output_strides: Box<[usize]> = output_layout.strides().into();
    let input_strides = input.layout().strides();
    let value_strides = value.layout().strides();
    let input_offset = input.layout().offset();
    let value_offset = value.layout().offset();

    ConcreteTensor::from_fn(input_shape, |out_linear| {
        // Convert linear index to multi-dimensional indices
        let mut indices = [0usize; R];
        let mut remaining = out_linear;
        for i in 0..R {
            indices[i] = remaining / output_strides[i];
            remaining %= output_strides[i];
        }

        // Check if this position is in the slice region
        if is_in_slice(&indices, slices) {
            // Get from value tensor (with stride calculation)
            let value_indices = to_value_indices(&indices, slices);
            let value_linear: usize = value_offset
                + value_indices
                    .iter()
                    .zip(value_strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum::<usize>();
            value.data()[value_linear]
        } else {
            // Copy from input tensor (with stride calculation)
            let input_linear: usize = input_offset
                + indices
                    .iter()
                    .zip(input_strides.iter())
                    .map(|(&idx, &stride)| idx * stride)
                    .sum::<usize>();
            input.data()[input_linear]
        }
    })
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_assign_2d_origin() {
        // 3x3 tensor with 2x2 value at [0..2, 0..2]
        let input = ConcreteTensor::<f32, 2>::from_slice(
            [3, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let value = ConcreteTensor::<f32, 2>::from_slice([2, 2], &[10.0, 11.0, 12.0, 13.0]);

        let result = slice_assign_ref(&input, [0..2, 0..2], &value);

        // Check slice region was replaced
        assert_eq!(result.get([0, 0]), 10.0);
        assert_eq!(result.get([0, 1]), 11.0);
        assert_eq!(result.get([1, 0]), 12.0);
        assert_eq!(result.get([1, 1]), 13.0);

        // Check rest was copied from input
        assert_eq!(result.get([0, 2]), 3.0);
        assert_eq!(result.get([1, 2]), 6.0);
        assert_eq!(result.get([2, 0]), 7.0);
        assert_eq!(result.get([2, 1]), 8.0);
        assert_eq!(result.get([2, 2]), 9.0);
    }

    #[test]
    fn test_slice_assign_2d_offset() {
        // 4x4 tensor with 2x2 value at [1..3, 2..4]
        let input = ConcreteTensor::<f32, 2>::from_slice(
            [4, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let value = ConcreteTensor::<f32, 2>::from_slice([2, 2], &[100.0, 101.0, 102.0, 103.0]);

        let result = slice_assign_ref(&input, [1..3, 2..4], &value);

        // Check slice region was replaced
        assert_eq!(result.get([1, 2]), 100.0);
        assert_eq!(result.get([1, 3]), 101.0);
        assert_eq!(result.get([2, 2]), 102.0);
        assert_eq!(result.get([2, 3]), 103.0);

        // Check surrounding values unchanged
        assert_eq!(result.get([0, 0]), 1.0);
        assert_eq!(result.get([1, 1]), 6.0);
        assert_eq!(result.get([3, 3]), 16.0);
    }

    #[test]
    fn test_slice_assign_1d() {
        let input = ConcreteTensor::<f32, 1>::from_slice([5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let value = ConcreteTensor::<f32, 1>::from_slice([2], &[10.0, 20.0]);

        let result = slice_assign_ref(&input, [1..3], &value);

        assert_eq!(result.get([0]), 1.0);
        assert_eq!(result.get([1]), 10.0);
        assert_eq!(result.get([2]), 20.0);
        assert_eq!(result.get([3]), 4.0);
        assert_eq!(result.get([4]), 5.0);
    }

    #[test]
    fn test_slice_assign_full_replacement() {
        // Replace entire 2x2 tensor
        let input = ConcreteTensor::<f32, 2>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let value = ConcreteTensor::<f32, 2>::from_slice([2, 2], &[10.0, 20.0, 30.0, 40.0]);

        let result = slice_assign_ref(&input, [0..2, 0..2], &value);

        assert_eq!(result.get([0, 0]), 10.0);
        assert_eq!(result.get([0, 1]), 20.0);
        assert_eq!(result.get([1, 0]), 30.0);
        assert_eq!(result.get([1, 1]), 40.0);
    }

    #[test]
    fn test_slice_assign_single_element() {
        let input = ConcreteTensor::<f32, 2>::from_slice([3, 3], &[1.0; 9]);
        let value = ConcreteTensor::<f32, 2>::from_slice([1, 1], &[99.0]);

        let result = slice_assign_ref(&input, [1..2, 1..2], &value);

        assert_eq!(result.get([1, 1]), 99.0);
        // All other elements should be 1.0
        assert_eq!(result.get([0, 0]), 1.0);
        assert_eq!(result.get([2, 2]), 1.0);
    }

    #[test]
    #[should_panic(expected = "Slice end")]
    fn test_slice_assign_out_of_bounds() {
        let input = ConcreteTensor::<f32, 2>::from_slice([3, 3], &[1.0; 9]);
        let value = ConcreteTensor::<f32, 2>::from_slice([2, 2], &[10.0; 4]);

        // Slice extends beyond tensor bounds
        let _ = slice_assign_ref(&input, [2..4, 0..2], &value);
    }

    #[test]
    #[should_panic(expected = "Value shape mismatch")]
    fn test_slice_assign_shape_mismatch() {
        let input = ConcreteTensor::<f32, 2>::from_slice([3, 3], &[1.0; 9]);
        let value = ConcreteTensor::<f32, 2>::from_slice([3, 2], &[10.0; 6]); // Wrong shape

        let _ = slice_assign_ref(&input, [0..2, 0..2], &value);
    }

    #[test]
    fn test_slice_assign_3d() {
        // 2x2x2 tensor
        let input = ConcreteTensor::<f32, 3>::from_slice(
            [2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let value = ConcreteTensor::<f32, 3>::from_slice([1, 1, 2], &[100.0, 200.0]);

        let result = slice_assign_ref(&input, [0..1, 1..2, 0..2], &value);

        // Check replaced values
        assert_eq!(result.get([0, 1, 0]), 100.0);
        assert_eq!(result.get([0, 1, 1]), 200.0);

        // Check unchanged values
        assert_eq!(result.get([0, 0, 0]), 1.0);
        assert_eq!(result.get([0, 0, 1]), 2.0);
        assert_eq!(result.get([1, 0, 0]), 5.0);
        assert_eq!(result.get([1, 1, 1]), 8.0);
    }
}
