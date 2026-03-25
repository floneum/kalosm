//! Tests for CPU-specific matrix multiplication behavior

use fusor_cpu::ConcreteTensor;

#[test]
#[should_panic(expected = "Matrix dimension mismatch")]
fn test_matmul_shape_mismatch() {
    let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &[1.0; 6]);
    let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 2], &[1.0; 4]);

    // This should panic because lhs columns (3) != rhs rows (2)
    let _ = lhs.matmul_ref(&rhs);
}
