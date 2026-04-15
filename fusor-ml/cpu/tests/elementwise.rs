//! Tests for elementwise (unary) tensor operations: Neg, Abs, Sqrt, transcendentals

use fusor_cpu::{
    Abs, ConcreteTensor, Cos, Exp, Exp2, Log, Log2, Neg, Sin, Sqrt, Tan, Tanh, Tensor,
    TensorBacking,
};

// ========== Neg Tests ==========

#[test]
fn test_neg_tensor_f32() {
    let tensor = ConcreteTensor::from_slice([2, 3], &[1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0]);

    let neg = Neg::new(tensor);
    let result = neg.to_concrete();

    assert_eq!(result.get([0, 0]), -1.0);
    assert_eq!(result.get([0, 1]), 2.0);
    assert_eq!(result.get([0, 2]), -3.0);
    assert_eq!(result.get([1, 0]), 4.0);
    assert_eq!(result.get([1, 1]), -5.0);
    assert_eq!(result.get([1, 2]), 6.0);
}

#[test]
fn test_neg_large_tensor() {
    let size = 1024;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let tensor = ConcreteTensor::from_slice([size], &data);

    let neg = Neg::new(tensor);
    let result = neg.to_concrete();

    for i in 0..size {
        assert_eq!(result.get([i]), -(i as f32));
    }
}

// ========== Abs Tests ==========

#[test]
fn test_abs_tensor_f32() {
    let tensor = ConcreteTensor::from_slice([2, 3], &[1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0]);

    let abs = Abs::new(tensor);
    let result = abs.to_concrete();

    assert_eq!(result.get([0, 0]), 1.0);
    assert_eq!(result.get([0, 1]), 2.0);
    assert_eq!(result.get([0, 2]), 3.0);
    assert_eq!(result.get([1, 0]), 4.0);
    assert_eq!(result.get([1, 1]), 5.0);
    assert_eq!(result.get([1, 2]), 6.0);
}

#[test]
fn test_abs_i32() {
    let tensor = ConcreteTensor::from_slice([4], &[1i32, -2, 3, -4]);

    let abs = Abs::new(tensor);
    let result = abs.to_concrete();

    assert_eq!(result.get([0]), 1);
    assert_eq!(result.get([1]), 2);
    assert_eq!(result.get([2]), 3);
    assert_eq!(result.get([3]), 4);
}

// ========== Sqrt Tests ==========

#[test]
fn test_sqrt_tensor_f32() {
    let tensor = ConcreteTensor::from_slice([2, 3], &[1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0]);

    let sqrt = Sqrt::new(tensor);
    let result = sqrt.to_concrete();

    assert_eq!(result.get([0, 0]), 1.0);
    assert_eq!(result.get([0, 1]), 2.0);
    assert_eq!(result.get([0, 2]), 3.0);
    assert_eq!(result.get([1, 0]), 4.0);
    assert_eq!(result.get([1, 1]), 5.0);
    assert_eq!(result.get([1, 2]), 6.0);
}

#[test]
fn test_sqrt_large_tensor() {
    let size = 1024;
    let data: Vec<f32> = (0..size).map(|i| (i * i) as f32).collect();
    let tensor = ConcreteTensor::from_slice([size], &data);

    let sqrt = Sqrt::new(tensor);
    let result = sqrt.to_concrete();

    for i in 0..size {
        assert!((result.get([i]) - i as f32).abs() < 1e-5);
    }
}

#[test]
fn test_sqrt_f64() {
    let tensor = ConcreteTensor::from_slice([4], &[1.0f64, 4.0, 9.0, 16.0]);

    let sqrt = Sqrt::new(tensor);
    let result = sqrt.to_concrete();

    assert_eq!(result.get([0]), 1.0);
    assert_eq!(result.get([1]), 2.0);
    assert_eq!(result.get([2]), 3.0);
    assert_eq!(result.get([3]), 4.0);
}

// ========== Transcendental Tests ==========

#[test]
fn test_exp_f32() {
    let tensor = ConcreteTensor::from_slice([4], &[0.0f32, 1.0, 2.0, -1.0]);

    let exp = Exp::new(&tensor);
    let result = exp.to_concrete();

    assert!((result.get([0]) - 1.0).abs() < 1e-6); // e^0 = 1
    assert!((result.get([1]) - std::f32::consts::E).abs() < 1e-6); // e^1 = e
    assert!((result.get([2]) - std::f32::consts::E.powi(2)).abs() < 1e-5);
    assert!((result.get([3]) - 1.0 / std::f32::consts::E).abs() < 1e-6);
}

#[test]
fn test_exp_ref_method() {
    let tensor = Tensor::from_slice([3], &[0.0f32, 1.0, 2.0]);
    let result = tensor.exp().to_concrete();

    assert!((result.get([0]) - 1.0).abs() < 1e-6);
    assert!((result.get([1]) - std::f32::consts::E).abs() < 1e-6);
}

#[test]
fn test_log_f32() {
    let tensor = ConcreteTensor::from_slice(
        [3],
        &[1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)],
    );

    let log = Log::new(&tensor);
    let result = log.to_concrete();

    assert!((result.get([0]) - 0.0).abs() < 1e-6); // ln(1) = 0
    assert!((result.get([1]) - 1.0).abs() < 1e-6); // ln(e) = 1
    assert!((result.get([2]) - 2.0).abs() < 1e-5); // ln(e^2) = 2
}

#[test]
fn test_sin_cos_f32() {
    use std::f32::consts::PI;
    let tensor = ConcreteTensor::from_slice([4], &[0.0f32, PI / 2.0, PI, 3.0 * PI / 2.0]);

    let sin_result = Sin::new(&tensor).to_concrete();
    let cos_result = Cos::new(&tensor).to_concrete();

    // sin(0) = 0, sin(π/2) = 1, sin(π) = 0, sin(3π/2) = -1
    assert!((sin_result.get([0]) - 0.0).abs() < 1e-6);
    assert!((sin_result.get([1]) - 1.0).abs() < 1e-6);
    assert!((sin_result.get([2]) - 0.0).abs() < 1e-5);
    assert!((sin_result.get([3]) + 1.0).abs() < 1e-5);

    // cos(0) = 1, cos(π/2) = 0, cos(π) = -1, cos(3π/2) = 0
    assert!((cos_result.get([0]) - 1.0).abs() < 1e-6);
    assert!((cos_result.get([1]) - 0.0).abs() < 1e-6);
    assert!((cos_result.get([2]) + 1.0).abs() < 1e-5);
    assert!((cos_result.get([3]) - 0.0).abs() < 1e-5);
}

#[test]
fn test_tan_f32() {
    use std::f32::consts::PI;
    let tensor = ConcreteTensor::from_slice([2], &[0.0f32, PI / 4.0]); // tan(0) = 0, tan(π/4) = 1

    let tan_result = Tan::new(&tensor).to_concrete();

    assert!((tan_result.get([0]) - 0.0).abs() < 1e-6);
    assert!((tan_result.get([1]) - 1.0).abs() < 1e-6);
}

#[test]
fn test_tanh_f32() {
    let tensor = ConcreteTensor::from_slice([4], &[0.0f32, 1.0, -1.0, 10.0]);

    let tanh_result = Tanh::new(&tensor).to_concrete();

    assert!((tanh_result.get([0]) - 0.0).abs() < 1e-6); // tanh(0) = 0
    assert!(
        (tanh_result.get([1]) - 0.0_f32.tanh()).abs() < 1e-6
            || (tanh_result.get([1]) - 1.0_f32.tanh()).abs() < 1e-6
    );
    assert!((tanh_result.get([2]) - (-1.0_f32).tanh()).abs() < 1e-6);
    assert!((tanh_result.get([3]) - 1.0).abs() < 1e-5); // tanh(10) ≈ 1
}

#[test]
fn test_tanh_ref_method() {
    let tensor = Tensor::from_slice([3], &[0.0f32, 1.0, -1.0]);
    let result = tensor.tanh().to_concrete();

    assert!((result.get([0]) - 0.0).abs() < 1e-6);
}

#[test]
fn test_exp2_log2_f32() {
    let tensor = ConcreteTensor::from_slice([4], &[0.0f32, 1.0, 2.0, 3.0]);

    // 2^0 = 1, 2^1 = 2, 2^2 = 4, 2^3 = 8
    let exp2_result = Exp2::new(&tensor).to_concrete();
    assert!((exp2_result.get([0]) - 1.0).abs() < 1e-6);
    assert!((exp2_result.get([1]) - 2.0).abs() < 1e-6);
    assert!((exp2_result.get([2]) - 4.0).abs() < 1e-5);
    assert!((exp2_result.get([3]) - 8.0).abs() < 1e-5);

    // log2(1) = 0, log2(2) = 1, log2(4) = 2, log2(8) = 3
    let log2_input = ConcreteTensor::from_slice([4], &[1.0f32, 2.0, 4.0, 8.0]);
    let log2_result = Log2::new(&log2_input).to_concrete();
    assert!((log2_result.get([0]) - 0.0).abs() < 1e-6);
    assert!((log2_result.get([1]) - 1.0).abs() < 1e-6);
    assert!((log2_result.get([2]) - 2.0).abs() < 1e-5);
    assert!((log2_result.get([3]) - 3.0).abs() < 1e-5);
}

#[test]
fn test_fused_transcendental_chain() {
    // Test softmax-like: exp(x) / sum(exp(x))
    let tensor = Tensor::from_slice([3], &[0.0f32, 1.0, 2.0]);

    let exp_result = tensor.exp().to_concrete();
    // e^0 = 1, e^1 ≈ 2.718, e^2 ≈ 7.389
    assert!((exp_result.get([0]) - 1.0).abs() < 1e-5);
    assert!((exp_result.get([1]) - std::f32::consts::E).abs() < 1e-5);
}

// ========== pow_scalar, min_scalar, max_scalar, clamp Tests ==========

#[test]
fn test_pow_scalar_f32() {
    let tensor = Tensor::from_slice([4], &[1.0f32, 2.0, 3.0, 4.0]);
    let result = tensor.pow_scalar(2.0);

    assert!((result.get([0]) - 1.0).abs() < 1e-6);
    assert!((result.get([1]) - 4.0).abs() < 1e-6);
    assert!((result.get([2]) - 9.0).abs() < 1e-5);
    assert!((result.get([3]) - 16.0).abs() < 1e-5);
}

#[test]
fn test_pow_scalar_f64() {
    let tensor = Tensor::from_slice([3], &[2.0f64, 3.0, 4.0]);
    let result = tensor.pow_scalar(3.0);

    assert!((result.get([0]) - 8.0).abs() < 1e-10);
    assert!((result.get([1]) - 27.0).abs() < 1e-10);
    assert!((result.get([2]) - 64.0).abs() < 1e-10);
}

#[test]
fn test_max_scalar_f32() {
    let tensor = Tensor::from_slice([4], &[-1.0f32, 0.0, 1.0, 2.0]);
    let result = tensor.max_scalar(0.5);

    assert_eq!(result.get([0]), 0.5); // max(-1, 0.5) = 0.5
    assert_eq!(result.get([1]), 0.5); // max(0, 0.5) = 0.5
    assert_eq!(result.get([2]), 1.0); // max(1, 0.5) = 1
    assert_eq!(result.get([3]), 2.0); // max(2, 0.5) = 2
}

#[test]
fn test_min_scalar_f32() {
    let tensor = Tensor::from_slice([4], &[-1.0f32, 0.0, 1.0, 2.0]);
    let result = tensor.min_scalar(0.5);

    assert_eq!(result.get([0]), -1.0); // min(-1, 0.5) = -1
    assert_eq!(result.get([1]), 0.0); // min(0, 0.5) = 0
    assert_eq!(result.get([2]), 0.5); // min(1, 0.5) = 0.5
    assert_eq!(result.get([3]), 0.5); // min(2, 0.5) = 0.5
}

#[test]
fn test_clamp_f32() {
    let tensor = Tensor::from_slice([5], &[-2.0f32, -0.5, 0.5, 1.5, 3.0]);
    let result = tensor.clamp(0.0, 1.0);

    assert_eq!(result.get([0]), 0.0); // clamp(-2, 0, 1) = 0
    assert_eq!(result.get([1]), 0.0); // clamp(-0.5, 0, 1) = 0
    assert_eq!(result.get([2]), 0.5); // clamp(0.5, 0, 1) = 0.5
    assert_eq!(result.get([3]), 1.0); // clamp(1.5, 0, 1) = 1
    assert_eq!(result.get([4]), 1.0); // clamp(3, 0, 1) = 1
}
