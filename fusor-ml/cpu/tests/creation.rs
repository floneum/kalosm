//! Tests for ConcreteTensor creation and basic operations

use fusor_cpu::{ConcreteTensor, ResolvedTensor};

#[test]
fn test_concrete_tensor_creation() {
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::zeros([2, 3]);
    assert_eq!(tensor.layout().shape(), &[2, 3]);
    assert_eq!(tensor.data().len(), 6);
    for i in 0..6 {
        assert_eq!(tensor.data()[i], 0.0);
    }
}

#[test]
fn test_concrete_tensor_from_slice() {
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let tensor: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &data);
    assert_eq!(tensor.layout().shape(), &[2, 3]);
    assert_eq!(tensor.get([0, 0]), 0.0);
    assert_eq!(tensor.get([0, 1]), 1.0);
    assert_eq!(tensor.get([0, 2]), 2.0);
    assert_eq!(tensor.get([1, 0]), 3.0);
    assert_eq!(tensor.get([1, 1]), 4.0);
    assert_eq!(tensor.get([1, 2]), 5.0);
}

#[test]
fn test_concrete_tensor_get_set() {
    let mut tensor: ConcreteTensor<f32, 2> = ConcreteTensor::zeros([2, 3]);
    tensor.set([0, 1], 42.0);
    tensor.set([1, 2], 100.0);
    assert_eq!(tensor.get([0, 1]), 42.0);
    assert_eq!(tensor.get([1, 2]), 100.0);
    assert_eq!(tensor.get([0, 0]), 0.0);
}
