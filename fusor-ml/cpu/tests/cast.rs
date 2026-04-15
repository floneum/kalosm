//! Tests for type casting operations

use fusor_cpu::{ConcreteTensor, Tensor};

#[test]
fn test_cast_f32_to_i32() {
    let a = Tensor::from_slice([4], &[1.5f32, 2.7, -3.2, 4.9]);
    let b: Tensor<1, ConcreteTensor<i32, 1>> = a.cast();

    assert_eq!(b.get([0]), 1); // 1.5 -> 1
    assert_eq!(b.get([1]), 2); // 2.7 -> 2
    assert_eq!(b.get([2]), -3); // -3.2 -> -3
    assert_eq!(b.get([3]), 4); // 4.9 -> 4
}

#[test]
fn test_cast_i32_to_f64() {
    let a = Tensor::from_slice([3], &[1i32, -2, 3]);
    let b: Tensor<1, ConcreteTensor<f64, 1>> = a.cast();

    assert_eq!(b.get([0]), 1.0);
    assert_eq!(b.get([1]), -2.0);
    assert_eq!(b.get([2]), 3.0);
}

#[test]
fn test_cast_f64_to_f32() {
    let a = Tensor::from_slice([3], &[1.5f64, 2.5, 3.5]);
    let b: Tensor<1, ConcreteTensor<f32, 1>> = a.cast();

    assert!((b.get([0]) - 1.5).abs() < 1e-6);
    assert!((b.get([1]) - 2.5).abs() < 1e-6);
    assert!((b.get([2]) - 3.5).abs() < 1e-6);
}

#[test]
fn test_cast_i32_to_i64() {
    let a = Tensor::from_slice([3], &[100i32, -200, 300]);
    let b: Tensor<1, ConcreteTensor<i64, 1>> = a.cast();

    assert_eq!(b.get([0]), 100);
    assert_eq!(b.get([1]), -200);
    assert_eq!(b.get([2]), 300);
}

#[test]
fn test_cast_u8_to_f32() {
    let a = Tensor::from_slice([4], &[0u8, 127, 200, 255]);
    let b: Tensor<1, ConcreteTensor<f32, 1>> = a.cast();

    assert_eq!(b.get([0]), 0.0);
    assert_eq!(b.get([1]), 127.0);
    assert_eq!(b.get([2]), 200.0);
    assert_eq!(b.get([3]), 255.0);
}

#[test]
fn test_cast_2d_tensor() {
    let a = Tensor::from_slice([2, 2], &[1.1f32, 2.2, 3.3, 4.4]);
    let b: Tensor<2, ConcreteTensor<i32, 2>> = a.cast();

    assert_eq!(b.get([0, 0]), 1);
    assert_eq!(b.get([0, 1]), 2);
    assert_eq!(b.get([1, 0]), 3);
    assert_eq!(b.get([1, 1]), 4);
}

#[test]
fn test_cast_same_type() {
    let a = Tensor::from_slice([3], &[1.0f32, 2.0, 3.0]);
    let b: Tensor<1, ConcreteTensor<f32, 1>> = a.cast();

    assert_eq!(b.get([0]), 1.0);
    assert_eq!(b.get([1]), 2.0);
    assert_eq!(b.get([2]), 3.0);
}

#[test]
fn test_cast_large() {
    let size = 1024;
    let data: Vec<f32> = (0..size).map(|i| i as f32 + 0.5).collect();
    let a = Tensor::from_slice([size], &data);
    let b: Tensor<1, ConcreteTensor<i32, 1>> = a.cast();

    for i in 0..size {
        assert_eq!(b.get([i]), i as i32);
    }
}
