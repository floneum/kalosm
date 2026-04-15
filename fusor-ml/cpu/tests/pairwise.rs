//! Tests for pairwise (binary) tensor operations: Add, Sub, Mul, Div

use fusor_cpu::{Add, ConcreteTensor, Div, Mul, Sub, TensorBacking};

// ========== Add Tests ==========

#[test]
fn test_add_tensor_contiguous() {
    let lhs_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &lhs_data);
    let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &rhs_data);

    let add = Add::new(lhs, rhs);
    let result = add.to_concrete();

    assert_eq!(result.layout().shape(), &[2, 3]);
    assert_eq!(result.get([0, 0]), 11.0);
    assert_eq!(result.get([0, 1]), 22.0);
    assert_eq!(result.get([0, 2]), 33.0);
    assert_eq!(result.get([1, 0]), 44.0);
    assert_eq!(result.get([1, 1]), 55.0);
    assert_eq!(result.get([1, 2]), 66.0);
}

#[test]
fn test_add_tensor_1d() {
    let lhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[1, 2, 3, 4]);
    let rhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[10, 20, 30, 40]);

    let add = Add::new(lhs, rhs);
    let result = add.to_concrete();

    assert_eq!(result.layout().shape(), &[4]);
    assert_eq!(result.get([0]), 11);
    assert_eq!(result.get([1]), 22);
    assert_eq!(result.get([2]), 33);
    assert_eq!(result.get([3]), 44);
}

#[test]
fn test_add_tensor_3d() {
    let lhs_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let rhs_data: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let lhs: ConcreteTensor<f64, 3> = ConcreteTensor::from_slice([2, 2, 2], &lhs_data);
    let rhs: ConcreteTensor<f64, 3> = ConcreteTensor::from_slice([2, 2, 2], &rhs_data);

    let add = Add::new(lhs, rhs);
    let result = add.to_concrete();

    assert_eq!(result.layout().shape(), &[2, 2, 2]);
    assert!((result.get([0, 0, 0]) - 1.1).abs() < 1e-10);
    assert!((result.get([0, 0, 1]) - 2.2).abs() < 1e-10);
    assert!((result.get([1, 1, 1]) - 8.8).abs() < 1e-10);
}

#[test]
fn test_add_large_tensor() {
    // Test with a larger tensor to exercise SIMD paths
    let size = 1024;
    let lhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

    let lhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &lhs_data);
    let rhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &rhs_data);

    let add = Add::new(lhs, rhs);
    let result = add.to_concrete();

    for i in 0..size {
        assert_eq!(result.get([i]), (i + i * 2) as f32);
    }
}

// ========== Sub Tests ==========

#[test]
fn test_sub_tensor_contiguous() {
    let lhs_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let rhs_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &lhs_data);
    let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &rhs_data);

    let sub = Sub::new(lhs, rhs);
    let result = sub.to_concrete();

    assert_eq!(result.layout().shape(), &[2, 3]);
    assert_eq!(result.get([0, 0]), 9.0);
    assert_eq!(result.get([0, 1]), 18.0);
    assert_eq!(result.get([0, 2]), 27.0);
    assert_eq!(result.get([1, 0]), 36.0);
    assert_eq!(result.get([1, 1]), 45.0);
    assert_eq!(result.get([1, 2]), 54.0);
}

#[test]
fn test_sub_large_tensor() {
    let size = 1024;
    let lhs_data: Vec<f32> = (0..size).map(|i| (i * 3) as f32).collect();
    let rhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    let lhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &lhs_data);
    let rhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &rhs_data);

    let sub = Sub::new(lhs, rhs);
    let result = sub.to_concrete();

    for i in 0..size {
        assert_eq!(result.get([i]), (i * 2) as f32);
    }
}

// ========== Mul Tests ==========

#[test]
fn test_mul_tensor_contiguous() {
    let lhs_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_data: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &lhs_data);
    let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &rhs_data);

    let mul = Mul::new(lhs, rhs);
    let result = mul.to_concrete();

    assert_eq!(result.layout().shape(), &[2, 3]);
    assert_eq!(result.get([0, 0]), 2.0);
    assert_eq!(result.get([0, 1]), 6.0);
    assert_eq!(result.get([0, 2]), 12.0);
    assert_eq!(result.get([1, 0]), 20.0);
    assert_eq!(result.get([1, 1]), 30.0);
    assert_eq!(result.get([1, 2]), 42.0);
}

#[test]
fn test_mul_large_tensor() {
    let size = 1024;
    let lhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let rhs_data: Vec<f32> = (0..size).map(|_| 2.0).collect();

    let lhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &lhs_data);
    let rhs: ConcreteTensor<f32, 1> = ConcreteTensor::from_slice([size], &rhs_data);

    let mul = Mul::new(lhs, rhs);
    let result = mul.to_concrete();

    for i in 0..size {
        assert_eq!(result.get([i]), (i * 2) as f32);
    }
}

#[test]
fn test_mul_i32() {
    let lhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[1, 2, 3, 4]);
    let rhs: ConcreteTensor<i32, 1> = ConcreteTensor::from_slice([4], &[10, 20, 30, 40]);

    let mul = Mul::new(lhs, rhs);
    let result = mul.to_concrete();

    assert_eq!(result.get([0]), 10);
    assert_eq!(result.get([1]), 40);
    assert_eq!(result.get([2]), 90);
    assert_eq!(result.get([3]), 160);
}

// ========== Div Tests ==========

#[test]
fn test_div_tensor_contiguous() {
    let lhs_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let rhs_data: Vec<f32> = vec![2.0, 4.0, 5.0, 8.0, 10.0, 12.0];

    let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &lhs_data);
    let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &rhs_data);

    let div = Div::new(lhs, rhs);
    let result = div.to_concrete();

    assert_eq!(result.layout().shape(), &[2, 3]);
    assert_eq!(result.get([0, 0]), 5.0);
    assert_eq!(result.get([0, 1]), 5.0);
    assert_eq!(result.get([0, 2]), 6.0);
    assert_eq!(result.get([1, 0]), 5.0);
    assert_eq!(result.get([1, 1]), 5.0);
    assert_eq!(result.get([1, 2]), 5.0);
}

#[test]
fn test_div_large_tensor() {
    let size = 1024;
    let lhs_data: Vec<f64> = (0..size).map(|i| (i * 4) as f64).collect();
    let rhs_data: Vec<f64> = (0..size).map(|_| 2.0).collect();

    let lhs: ConcreteTensor<f64, 1> = ConcreteTensor::from_slice([size], &lhs_data);
    let rhs: ConcreteTensor<f64, 1> = ConcreteTensor::from_slice([size], &rhs_data);

    let div = Div::new(lhs, rhs);
    let result = div.to_concrete();

    for i in 0..size {
        assert_eq!(result.get([i]), (i * 2) as f64);
    }
}
