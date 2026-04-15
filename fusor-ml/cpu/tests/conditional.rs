//! Tests for conditional tensor operations: where_cond

use fusor_cpu::Tensor;

#[test]
fn test_where_cond_basic_f32() {
    let cond = Tensor::from_slice([4], &[1.0f32, 0.0, 1.0, 0.0]);
    let on_true = Tensor::from_slice([4], &[10.0f32, 20.0, 30.0, 40.0]);
    let on_false = Tensor::from_slice([4], &[100.0f32, 200.0, 300.0, 400.0]);

    let result = cond.where_cond(on_true, on_false);

    assert_eq!(result.get([0]), 10.0); // cond=1 (nonzero), select on_true
    assert_eq!(result.get([1]), 200.0); // cond=0, select on_false
    assert_eq!(result.get([2]), 30.0); // cond=1 (nonzero), select on_true
    assert_eq!(result.get([3]), 400.0); // cond=0, select on_false
}

#[test]
fn test_where_cond_nonzero_values_f32() {
    // Test that any nonzero value is truthy
    let cond = Tensor::from_slice([4], &[0.5f32, -1.0, 0.0, 100.0]);
    let on_true = Tensor::from_slice([4], &[1.0f32, 1.0, 1.0, 1.0]);
    let on_false = Tensor::from_slice([4], &[0.0f32, 0.0, 0.0, 0.0]);

    let result = cond.where_cond(on_true, on_false);

    assert_eq!(result.get([0]), 1.0); // 0.5 != 0
    assert_eq!(result.get([1]), 1.0); // -1.0 != 0
    assert_eq!(result.get([2]), 0.0); // 0.0 == 0
    assert_eq!(result.get([3]), 1.0); // 100.0 != 0
}

#[test]
fn test_where_cond_i32() {
    let cond = Tensor::from_slice([4], &[1i32, 0, -1, 0]);
    let on_true = Tensor::from_slice([4], &[10i32, 20, 30, 40]);
    let on_false = Tensor::from_slice([4], &[100i32, 200, 300, 400]);

    let result = cond.where_cond(on_true, on_false);

    assert_eq!(result.get([0]), 10); // 1 != 0
    assert_eq!(result.get([1]), 200); // 0 == 0
    assert_eq!(result.get([2]), 30); // -1 != 0
    assert_eq!(result.get([3]), 400); // 0 == 0
}

#[test]
fn test_where_cond_2d() {
    let cond = Tensor::from_slice([2, 2], &[1.0f32, 0.0, 0.0, 1.0]);
    let on_true = Tensor::from_slice([2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
    let on_false = Tensor::from_slice([2, 2], &[10.0f32, 20.0, 30.0, 40.0]);

    let result = cond.where_cond(on_true, on_false);

    assert_eq!(result.get([0, 0]), 1.0); // cond=1
    assert_eq!(result.get([0, 1]), 20.0); // cond=0
    assert_eq!(result.get([1, 0]), 30.0); // cond=0
    assert_eq!(result.get([1, 1]), 4.0); // cond=1
}

#[test]
fn test_where_cond_with_comparison() {
    // Common pattern: where_cond with comparison result
    let a = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([4], &[2.5f32, 2.5, 2.5, 2.5]);
    let a2 = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let b2 = Tensor::from_slice([4], &[2.5f32, 2.5, 2.5, 2.5]);

    // Compute (a > b) as mask and materialize it
    let mask = a.gt(b).to_concrete();
    // mask = [0, 0, 1, 1]

    // Use mask to select: where a > b, return a, else return b
    let result = Tensor::new(mask).where_cond(a2, b2);

    assert_eq!(result.get([0]), 2.5); // 1 > 2.5 = false, select b
    assert_eq!(result.get([1]), 2.5); // 2 > 2.5 = false, select b
    assert_eq!(result.get([2]), 3.0); // 3 > 2.5 = true, select a
    assert_eq!(result.get([3]), 4.0); // 4 > 2.5 = true, select a
}

#[test]
fn test_where_cond_large() {
    let size = 1024;
    let cond_data: Vec<f32> = (0..size)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let true_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let false_data: Vec<f32> = (0..size).map(|i| -(i as f32)).collect();

    let cond = Tensor::from_slice([size], &cond_data);
    let on_true = Tensor::from_slice([size], &true_data);
    let on_false = Tensor::from_slice([size], &false_data);

    let result = cond.where_cond(on_true, on_false);

    for i in 0..size {
        if i % 2 == 0 {
            assert_eq!(result.get([i]), i as f32);
        } else {
            assert_eq!(result.get([i]), -(i as f32));
        }
    }
}
