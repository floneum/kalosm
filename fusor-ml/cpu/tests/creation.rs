//! Tests for backend-specific ConcreteTensor operations

use fusor_cpu::ConcreteTensor;

#[test]
fn test_concrete_tensor_get_set() {
    let mut tensor: ConcreteTensor<f32, 2> = ConcreteTensor::zeros([2, 3]);
    tensor.set([0, 1], 42.0);
    tensor.set([1, 2], 100.0);
    assert_eq!(tensor.get([0, 1]), 42.0);
    assert_eq!(tensor.get([1, 2]), 100.0);
    assert_eq!(tensor.get([0, 0]), 0.0);
}
