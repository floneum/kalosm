//! Tests for index operations: index_select

use fusor_cpu::Tensor;

#[test]
fn test_index_select_dim0_2d() {
    // [[1, 2, 3], [4, 5, 6]] with indices [1, 0] -> [[4, 5, 6], [1, 2, 3]]
    let input = Tensor::from_slice([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let indices = Tensor::from_slice([2], &[1u32, 0]);

    let result = input.index_select(0, indices);

    assert_eq!(result.get([0, 0]), 4.0);
    assert_eq!(result.get([0, 1]), 5.0);
    assert_eq!(result.get([0, 2]), 6.0);
    assert_eq!(result.get([1, 0]), 1.0);
    assert_eq!(result.get([1, 1]), 2.0);
    assert_eq!(result.get([1, 2]), 3.0);
}

#[test]
fn test_index_select_dim1_2d() {
    // [[1, 2, 3], [4, 5, 6]] with indices [1, 2, 0] -> [[2, 3, 1], [5, 6, 4]]
    let input = Tensor::from_slice([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let indices = Tensor::from_slice([3], &[1u32, 2, 0]);

    let result = input.index_select(1, indices);

    assert_eq!(result.get([0, 0]), 2.0);
    assert_eq!(result.get([0, 1]), 3.0);
    assert_eq!(result.get([0, 2]), 1.0);
    assert_eq!(result.get([1, 0]), 5.0);
    assert_eq!(result.get([1, 1]), 6.0);
    assert_eq!(result.get([1, 2]), 4.0);
}

#[test]
fn test_index_select_1d() {
    let input = Tensor::from_slice([5], &[10.0f32, 20.0, 30.0, 40.0, 50.0]);
    let indices = Tensor::from_slice([3], &[4u32, 2, 0]);

    let result = input.index_select(0, indices);

    assert_eq!(result.get([0]), 50.0);
    assert_eq!(result.get([1]), 30.0);
    assert_eq!(result.get([2]), 10.0);
}

#[test]
fn test_index_select_duplicate_indices() {
    // Select same row multiple times
    let input = Tensor::from_slice([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let indices = Tensor::from_slice([4], &[0u32, 0, 1, 1]);

    let result = input.index_select(0, indices);

    // Should be [[1,2,3], [1,2,3], [4,5,6], [4,5,6]]
    assert_eq!(result.get([0, 0]), 1.0);
    assert_eq!(result.get([1, 0]), 1.0);
    assert_eq!(result.get([2, 0]), 4.0);
    assert_eq!(result.get([3, 0]), 4.0);
}

#[test]
fn test_index_select_single_index() {
    let input = Tensor::from_slice([3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let indices = Tensor::from_slice([1], &[1u32]);

    let result = input.index_select(0, indices);

    // Should select row 1: [3, 4]
    assert_eq!(result.get([0, 0]), 3.0);
    assert_eq!(result.get([0, 1]), 4.0);
}

#[test]
fn test_index_select_i32() {
    let input = Tensor::from_slice([4], &[100i32, 200, 300, 400]);
    let indices = Tensor::from_slice([2], &[3u32, 1]);

    let result = input.index_select(0, indices);

    assert_eq!(result.get([0]), 400);
    assert_eq!(result.get([1]), 200);
}

#[test]
fn test_index_select_3d() {
    // Shape [2, 2, 2] tensor
    let input = Tensor::from_slice([2, 2, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let indices = Tensor::from_slice([2], &[1u32, 0]);

    // Select along dim 0 (swap the two 2x2 matrices)
    let result = input.index_select(0, indices);

    // Original: [[[1,2],[3,4]], [[5,6],[7,8]]]
    // After: [[[5,6],[7,8]], [[1,2],[3,4]]]
    assert_eq!(result.get([0, 0, 0]), 5.0);
    assert_eq!(result.get([0, 0, 1]), 6.0);
    assert_eq!(result.get([0, 1, 0]), 7.0);
    assert_eq!(result.get([0, 1, 1]), 8.0);
    assert_eq!(result.get([1, 0, 0]), 1.0);
    assert_eq!(result.get([1, 0, 1]), 2.0);
}

#[test]
fn test_index_select_large() {
    let size = 100;
    let data: Vec<f32> = (0..size * size).map(|i| i as f32).collect();
    let input = Tensor::from_slice([size, size], &data);

    // Reverse the rows
    let indices_data: Vec<u32> = (0..size).rev().map(|i| i as u32).collect();
    let indices = Tensor::from_slice([size], &indices_data);

    let result = input.index_select(0, indices);

    // Check that rows are reversed
    for i in 0..size {
        let expected_row = size - 1 - i;
        assert_eq!(result.get([i, 0]), (expected_row * size) as f32);
    }
}
