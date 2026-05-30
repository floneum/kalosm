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
