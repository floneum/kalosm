//! Tests for matrix multiplication operations

use fusor_cpu::ConcreteTensor;

#[test]
fn test_matmul_2x3_3x2() {
    // [1 2 3]   [1 2]   [22 28]
    // [4 5 6] @ [3 4] = [49 64]
    //           [5 6]
    let lhs: ConcreteTensor<f32, 2> =
        ConcreteTensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs: ConcreteTensor<f32, 2> =
        ConcreteTensor::from_slice([3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = lhs.matmul_ref(&rhs);

    assert_eq!(result.layout().shape(), &[2, 2]);
    assert_eq!(result.get([0, 0]), 22.0);
    assert_eq!(result.get([0, 1]), 28.0);
    assert_eq!(result.get([1, 0]), 49.0);
    assert_eq!(result.get([1, 1]), 64.0);
}

#[test]
fn test_matmul_identity() {
    // Matrix times identity should return the original matrix
    let mat: ConcreteTensor<f32, 2> =
        ConcreteTensor::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let identity: ConcreteTensor<f32, 2> =
        ConcreteTensor::from_slice([3, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    let result = mat.matmul_ref(&identity);

    assert_eq!(result.layout().shape(), &[2, 3]);
    assert_eq!(result.get([0, 0]), 1.0);
    assert_eq!(result.get([0, 1]), 2.0);
    assert_eq!(result.get([0, 2]), 3.0);
    assert_eq!(result.get([1, 0]), 4.0);
    assert_eq!(result.get([1, 1]), 5.0);
    assert_eq!(result.get([1, 2]), 6.0);
}

#[test]
fn test_matmul_large() {
    // Test with larger matrices to exercise the gemm path
    let size = 64;
    let lhs_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
    let rhs_data: Vec<f32> = (0..size * size).map(|i| ((i + 1) % 10) as f32).collect();

    let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([size, size], &lhs_data);
    let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([size, size], &rhs_data);

    let result = lhs.matmul_ref(&rhs);

    assert_eq!(result.layout().shape(), &[size, size]);

    // Verify a few elements by computing them manually
    // result[0,0] = sum(lhs[0,:] * rhs[:,0])
    let mut expected_00: f32 = 0.0;
    for k in 0..size {
        expected_00 += lhs_data[k] * rhs_data[k * size];
    }
    assert!((result.get([0, 0]) - expected_00).abs() < 1e-3);
}

#[test]
fn test_matmul_f64() {
    // Test f64 path
    let lhs: ConcreteTensor<f64, 2> = ConcreteTensor::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let rhs: ConcreteTensor<f64, 2> = ConcreteTensor::from_slice([2, 2], &[5.0, 6.0, 7.0, 8.0]);

    let result = lhs.matmul_ref(&rhs);

    assert_eq!(result.layout().shape(), &[2, 2]);
    // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    assert_eq!(result.get([0, 0]), 19.0);
    assert_eq!(result.get([0, 1]), 22.0);
    assert_eq!(result.get([1, 0]), 43.0);
    assert_eq!(result.get([1, 1]), 50.0);
}

#[test]
#[should_panic(expected = "Matrix dimension mismatch")]
fn test_matmul_shape_mismatch() {
    let lhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 3], &[1.0; 6]);
    let rhs: ConcreteTensor<f32, 2> = ConcreteTensor::from_slice([2, 2], &[1.0; 4]);

    // This should panic because lhs columns (3) != rhs rows (2)
    let _ = lhs.matmul_ref(&rhs);
}
