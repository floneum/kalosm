//! Tests for comparison tensor operations: eq, ne, lt, lte, gt, gte

use fusor_cpu::Tensor;

// ========== Tensor vs Tensor Tests ==========

#[test]
fn test_eq_tensor_f32() {
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([4], &[1.0f32, 3.0, 3.0, 5.0]);

    let result = a.eq(b).to_concrete();

    assert_eq!(result.get([0]), 1.0); // 1 == 1
    assert_eq!(result.get([1]), 0.0); // 2 != 3
    assert_eq!(result.get([2]), 1.0); // 3 == 3
    assert_eq!(result.get([3]), 0.0); // 4 != 5
}

#[test]
fn test_ne_tensor_f32() {
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([4], &[1.0f32, 3.0, 3.0, 5.0]);

    let result = a.ne(b).to_concrete();

    assert_eq!(result.get([0]), 0.0); // 1 == 1 -> false
    assert_eq!(result.get([1]), 1.0); // 2 != 3 -> true
    assert_eq!(result.get([2]), 0.0); // 3 == 3 -> false
    assert_eq!(result.get([3]), 1.0); // 4 != 5 -> true
}

#[test]
fn test_lt_tensor_f32() {
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([4], &[2.0f32, 2.0, 2.0, 2.0]);

    let result = a.lt(b).to_concrete();

    assert_eq!(result.get([0]), 1.0); // 1 < 2
    assert_eq!(result.get([1]), 0.0); // 2 < 2
    assert_eq!(result.get([2]), 0.0); // 3 < 2
    assert_eq!(result.get([3]), 0.0); // 4 < 2
}

#[test]
fn test_lte_tensor_f32() {
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([4], &[2.0f32, 2.0, 2.0, 2.0]);

    let result = a.lte(b).to_concrete();

    assert_eq!(result.get([0]), 1.0); // 1 <= 2
    assert_eq!(result.get([1]), 1.0); // 2 <= 2
    assert_eq!(result.get([2]), 0.0); // 3 <= 2
    assert_eq!(result.get([3]), 0.0); // 4 <= 2
}

#[test]
fn test_gt_tensor_f32() {
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([4], &[2.0f32, 2.0, 2.0, 2.0]);

    let result = a.gt(b).to_concrete();

    assert_eq!(result.get([0]), 0.0); // 1 > 2
    assert_eq!(result.get([1]), 0.0); // 2 > 2
    assert_eq!(result.get([2]), 1.0); // 3 > 2
    assert_eq!(result.get([3]), 1.0); // 4 > 2
}

#[test]
fn test_gte_tensor_f32() {
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([4], &[2.0f32, 2.0, 2.0, 2.0]);

    let result = a.gte(b).to_concrete();

    assert_eq!(result.get([0]), 0.0); // 1 >= 2
    assert_eq!(result.get([1]), 1.0); // 2 >= 2
    assert_eq!(result.get([2]), 1.0); // 3 >= 2
    assert_eq!(result.get([3]), 1.0); // 4 >= 2
}

// ========== Tensor vs Scalar Tests ==========

#[test]
fn test_eq_scalar_f32() {
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 2.0, 4.0]);
    let result = a.eq_scalar(2.0).to_concrete();

    assert_eq!(result.get([0]), 0.0);
    assert_eq!(result.get([1]), 1.0);
    assert_eq!(result.get([2]), 1.0);
    assert_eq!(result.get([3]), 0.0);
}

#[test]
fn test_lt_scalar_f32() {
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let result = a.lt_scalar(2.5).to_concrete();

    assert_eq!(result.get([0]), 1.0); // 1 < 2.5
    assert_eq!(result.get([1]), 1.0); // 2 < 2.5
    assert_eq!(result.get([2]), 0.0); // 3 < 2.5
    assert_eq!(result.get([3]), 0.0); // 4 < 2.5
}

#[test]
fn test_gt_scalar_f32() {
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let result = a.gt_scalar(2.5).to_concrete();

    assert_eq!(result.get([0]), 0.0); // 1 > 2.5
    assert_eq!(result.get([1]), 0.0); // 2 > 2.5
    assert_eq!(result.get([2]), 1.0); // 3 > 2.5
    assert_eq!(result.get([3]), 1.0); // 4 > 2.5
}

// ========== Integer Tests ==========

#[test]
fn test_comparison_i32() {
    let a = Tensor::from_slice([4], &[1i32, 2, 3, 4]);
    let b = Tensor::from_slice([4], &[2i32, 2, 2, 2]);

    let lt_result = a.as_ref().lt(b).to_concrete();
    assert_eq!(lt_result.get([0]), 1); // 1 < 2
    assert_eq!(lt_result.get([1]), 0); // 2 < 2
    assert_eq!(lt_result.get([2]), 0); // 3 < 2
    assert_eq!(lt_result.get([3]), 0); // 4 < 2

    let eq_result = a.eq_scalar(2).to_concrete();
    assert_eq!(eq_result.get([0]), 0);
    assert_eq!(eq_result.get([1]), 1);
    assert_eq!(eq_result.get([2]), 0);
    assert_eq!(eq_result.get([3]), 0);
}

// ========== Large Tensor Tests ==========

#[test]
fn test_comparison_large_tensor() {
    let size = 1024;
    let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size).map(|_| (size / 2) as f32).collect();

    let a = Tensor::from_slice([size], &a_data);
    let b = Tensor::from_slice([size], &b_data);

    let result = a.lt(b).to_concrete();

    for i in 0..size {
        let expected = if (i as f32) < (size / 2) as f32 {
            1.0
        } else {
            0.0
        };
        assert_eq!(result.get([i]), expected);
    }
}
