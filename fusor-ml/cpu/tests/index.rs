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
fn test_sliding_window_transpose_reshape() {
    use fusor_types::SlidingWindow;

    // Simulate what conv1d does:
    // Input: (1, in_ch=2, length=5)
    // After sliding_window: (1, in_ch=2, out_len=3, kernel=3) (with step=1, kernel_size=3)
    // After transpose(1,2): (1, out_len=3, in_ch=2, kernel=3)
    // After reshape: (3, 6)

    let input_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    // Layout: [[0,1,2,3,4], [5,6,7,8,9]]
    let input = Tensor::from_slice([1, 2, 5], &input_data);

    // Sliding window on axis 2 with window_size=3, step=1
    // Produces (1, 2, 3, 3) where last dim is window content
    let windows: Tensor<4, _> = input.sliding_window_view([SlidingWindow::new(2, 3, 1)]);

    // Transpose axes 1 and 2 (swap in_channels and out_length)
    let transposed = windows.transpose(1, 2);

    // Reshape to (3, 6) - flatten batch*out_len and in_ch*kernel
    let reshaped: Tensor<2, _> = transposed.reshape([3, 6]);

    // Expected values:
    // Window 0 for channel 0: [0, 1, 2]
    // Window 0 for channel 1: [5, 6, 7]
    // So row 0 should be [0, 1, 2, 5, 6, 7]
    //
    // Window 1 for channel 0: [1, 2, 3]
    // Window 1 for channel 1: [6, 7, 8]
    // So row 1 should be [1, 2, 3, 6, 7, 8]
    //
    // Window 2 for channel 0: [2, 3, 4]
    // Window 2 for channel 1: [7, 8, 9]
    // So row 2 should be [2, 3, 4, 7, 8, 9]

    assert_eq!(reshaped.get([0, 0]), 0.0, "row 0, col 0");
    assert_eq!(reshaped.get([0, 1]), 1.0, "row 0, col 1");
    assert_eq!(reshaped.get([0, 2]), 2.0, "row 0, col 2");
    assert_eq!(reshaped.get([0, 3]), 5.0, "row 0, col 3");
    assert_eq!(reshaped.get([0, 4]), 6.0, "row 0, col 4");
    assert_eq!(reshaped.get([0, 5]), 7.0, "row 0, col 5");

    assert_eq!(reshaped.get([1, 0]), 1.0, "row 1, col 0");
    assert_eq!(reshaped.get([1, 1]), 2.0, "row 1, col 1");
    assert_eq!(reshaped.get([1, 2]), 3.0, "row 1, col 2");
    assert_eq!(reshaped.get([1, 3]), 6.0, "row 1, col 3");
    assert_eq!(reshaped.get([1, 4]), 7.0, "row 1, col 4");
    assert_eq!(reshaped.get([1, 5]), 8.0, "row 1, col 5");

    assert_eq!(reshaped.get([2, 0]), 2.0, "row 2, col 0");
    assert_eq!(reshaped.get([2, 1]), 3.0, "row 2, col 1");
    assert_eq!(reshaped.get([2, 2]), 4.0, "row 2, col 2");
    assert_eq!(reshaped.get([2, 3]), 7.0, "row 2, col 3");
    assert_eq!(reshaped.get([2, 4]), 8.0, "row 2, col 4");
    assert_eq!(reshaped.get([2, 5]), 9.0, "row 2, col 5");
}

#[test]
fn test_sliding_window_with_cat_padding() {
    use fusor_types::SlidingWindow;

    // Simulate what conv1d does with padding:
    // Input: (1, in_ch=2, length=3)
    // After padding with cat: (1, in_ch=2, length=5) [pad, val, val, val, pad]
    // After sliding_window: (1, in_ch=2, out_len=3, kernel=3)
    // After transpose(1,2): (1, out_len=3, in_ch=2, kernel=3)
    // After reshape: (3, 6)

    // Original data: channel 0 = [1, 2, 3], channel 1 = [4, 5, 6]
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_slice([1, 2, 3], &input_data);

    // Manually pad by concatenating
    use fusor_cpu::ConcreteTensor;
    let pad_left: Tensor<3, ConcreteTensor<f32, 3>> = Tensor::zeros([1, 2, 1]);
    let pad_right: Tensor<3, ConcreteTensor<f32, 3>> = Tensor::zeros([1, 2, 1]);
    let padded = Tensor::cat([pad_left, input, pad_right], 2);

    // Verify padding worked:
    // Padded should be: channel 0 = [0, 1, 2, 3, 0], channel 1 = [0, 4, 5, 6, 0]
    assert_eq!(padded.get([0, 0, 0]), 0.0, "padded ch0 pos0");
    assert_eq!(padded.get([0, 0, 1]), 1.0, "padded ch0 pos1");
    assert_eq!(padded.get([0, 0, 2]), 2.0, "padded ch0 pos2");
    assert_eq!(padded.get([0, 0, 3]), 3.0, "padded ch0 pos3");
    assert_eq!(padded.get([0, 0, 4]), 0.0, "padded ch0 pos4");
    assert_eq!(padded.get([0, 1, 0]), 0.0, "padded ch1 pos0");
    assert_eq!(padded.get([0, 1, 1]), 4.0, "padded ch1 pos1");

    // Sliding window on axis 2
    let windows: Tensor<4, _> = padded.sliding_window_view([SlidingWindow::new(2, 3, 1)]);

    // Transpose axes 1 and 2
    let transposed = windows.transpose(1, 2);

    // Reshape to (3, 6)
    let reshaped: Tensor<2, _> = transposed.reshape([3, 6]);

    // Expected values:
    // Window 0 for channel 0: [0, 1, 2]
    // Window 0 for channel 1: [0, 4, 5]
    // So row 0 should be [0, 1, 2, 0, 4, 5]
    //
    // Window 1 for channel 0: [1, 2, 3]
    // Window 1 for channel 1: [4, 5, 6]
    // So row 1 should be [1, 2, 3, 4, 5, 6]
    //
    // Window 2 for channel 0: [2, 3, 0]
    // Window 2 for channel 1: [5, 6, 0]
    // So row 2 should be [2, 3, 0, 5, 6, 0]

    println!(
        "Row 0: {:?}",
        (0..6).map(|i| reshaped.get([0, i])).collect::<Vec<_>>()
    );
    println!(
        "Row 1: {:?}",
        (0..6).map(|i| reshaped.get([1, i])).collect::<Vec<_>>()
    );
    println!(
        "Row 2: {:?}",
        (0..6).map(|i| reshaped.get([2, i])).collect::<Vec<_>>()
    );

    assert_eq!(reshaped.get([0, 0]), 0.0, "row 0, col 0 (pad)");
    assert_eq!(reshaped.get([0, 1]), 1.0, "row 0, col 1");
    assert_eq!(reshaped.get([0, 2]), 2.0, "row 0, col 2");
    assert_eq!(reshaped.get([0, 3]), 0.0, "row 0, col 3 (pad)");
    assert_eq!(reshaped.get([0, 4]), 4.0, "row 0, col 4");
    assert_eq!(reshaped.get([0, 5]), 5.0, "row 0, col 5");

    assert_eq!(reshaped.get([1, 0]), 1.0, "row 1, col 0");
    assert_eq!(reshaped.get([1, 1]), 2.0, "row 1, col 1");
    assert_eq!(reshaped.get([1, 2]), 3.0, "row 1, col 2");
    assert_eq!(reshaped.get([1, 3]), 4.0, "row 1, col 3");
    assert_eq!(reshaped.get([1, 4]), 5.0, "row 1, col 4");
    assert_eq!(reshaped.get([1, 5]), 6.0, "row 1, col 5");

    assert_eq!(reshaped.get([2, 0]), 2.0, "row 2, col 0");
    assert_eq!(reshaped.get([2, 1]), 3.0, "row 2, col 1");
    assert_eq!(reshaped.get([2, 2]), 0.0, "row 2, col 2 (pad)");
    assert_eq!(reshaped.get([2, 3]), 5.0, "row 2, col 3");
    assert_eq!(reshaped.get([2, 4]), 6.0, "row 2, col 4");
    assert_eq!(reshaped.get([2, 5]), 0.0, "row 2, col 5 (pad)");
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
