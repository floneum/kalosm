//! Activation functions that work on both CPU and GPU backends.

use crate::{
    AddOp, DivOp, ExpOp, FloatOps, MulOp, NegOp, SimdBinaryOp, SimdElement, SimdUnaryOp, TanhOp,
    Tensor,
};
use fusor_core::{DataType, FloatDataType};

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Rectified Linear Unit activation: relu(x) = max(0, x)
    pub fn relu(&self) -> Self {
        self.max_scalar(D::from_f32(0.0))
    }

    /// Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor<R, D>
    where
        D: std::ops::Neg<Output = D>
            + std::ops::Add<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Mul<Output = D>
            + fusor_cpu::Scalar,
        AddOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        NegOp: SimdUnaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        let neg_self = -self;
        let exp_neg = neg_self.exp();
        let one_plus_exp = (exp_neg + D::from_f32(1.0)).to_concrete();
        // Create ones with the same shape by: x * 0 + 1
        let ones = (self.mul_scalar(D::from_f32(0.0)) + D::from_f32(1.0)).to_concrete();
        (ones / one_plus_exp).to_concrete()
    }

    /// Sigmoid Linear Unit activation: silu(x) = x * sigmoid(x)
    pub fn silu(&self) -> Tensor<R, D>
    where
        D: std::ops::Neg<Output = D>
            + std::ops::Add<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Mul<Output = D>
            + fusor_cpu::Scalar,
        AddOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        NegOp: SimdUnaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let neg_self = -self;
        let exp_neg = neg_self.exp();
        let one_plus_exp = exp_neg + D::from_f32(1.0);
        (self / one_plus_exp).to_concrete()
    }

    /// Gaussian Error Linear Unit activation (approximate).
    ///
    /// Uses the tanh approximation:
    /// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Self
    where
        AddOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        TanhOp: SimdUnaryOp<D>,
        D: fusor_cpu::Scalar,
    {
        let coeff = D::from_f32((2.0 / std::f32::consts::PI).sqrt());

        // x^2
        let x_squared = self * self;

        // 0.044715 * x^2 + 1.0
        let inner_factor = x_squared * D::from_f32(0.044715) + D::from_f32(1.0);

        // x * (1 + 0.044715 * x^2)
        let inner = self * &inner_factor;

        // sqrt(2/pi) * (x * (1 + 0.044715 * x^2))
        let tanh_input = inner * coeff;

        // Clamp tanh INPUT to [-15, 15] to prevent GPU NaN from WGSL tanh overflow.
        // WGSL's tanh(x) computes (e^x - e^-x)/(e^x + e^-x), but e^x overflows f32
        // for x > ~88. For |x| > 10, tanh(x) ≈ ±1.0, so clamping to ±15 is safe.
        let tanh_input = tanh_input.clamp(D::from_f32(-15.0), D::from_f32(15.0));
        let tanh_result = tanh_input.tanh();

        // 1 + tanh(...)
        let one_plus_tanh = &tanh_result + D::from_f32(1.0);

        // x * (1 + tanh(...))
        let product = self * &one_plus_tanh;

        // 0.5 * x * (1 + tanh(...))
        (product * D::from_f32(0.5)).to_concrete()
    }
}

#[cfg(test)]
#[allow(clippy::identity_op, clippy::useless_conversion)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_relu_cpu() {
        let t: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [6],
            &[1.0, -2.0, -3.0, 4.0, 5.0, -6.0],
        ));
        let result = t.relu();
        let slice = result.as_slice().await.unwrap();

        assert!((slice[[0]] - 1.0).abs() < 0.001);
        assert!((slice[[1]] - 0.0).abs() < 0.001);
        assert!((slice[[2]] - 0.0).abs() < 0.001);
        assert!((slice[[3]] - 4.0).abs() < 0.001);
        assert!((slice[[4]] - 5.0).abs() < 0.001);
        assert!((slice[[5]] - 0.0).abs() < 0.001);
    }

    fn silu_ref(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    #[tokio::test]
    async fn test_silu_cpu() {
        let data = [1.0f32, -2.0, -3.0, 4.0, 5.0, -6.0];
        let t: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([6], &data));
        let result = t.silu();
        let slice = result.as_slice().await.unwrap();

        for i in 0..6 {
            assert!(
                (slice[[i]] - silu_ref(data[i])).abs() < 0.001,
                "Mismatch at index {}",
                i
            );
        }
    }

    fn gelu_ref(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    #[tokio::test]
    async fn test_gelu_cpu() {
        let data = [1.0f32, -2.0, -3.0, 4.0, 5.0, -5.0];
        let t: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([6], &data));
        let result = t.gelu();
        let slice = result.as_slice().await.unwrap();

        for i in 0..6 {
            assert!(
                (slice[[i]] - gelu_ref(data[i])).abs() < 0.01,
                "Mismatch at index {}: got {}, expected {}",
                i,
                slice[[i]],
                gelu_ref(data[i])
            );
        }
    }

    #[tokio::test]
    async fn test_gelu_large_values() {
        use crate::Device;

        // Test GELU for large values where intermediate tanh_input overflows f32
        let data = [10.0f32, 100.0, 500.0, 1000.0, 2725.0, -10.0, -100.0, -500.0];

        // GPU version
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let gpu_tensor: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [8], &data);
        let gpu_result = gpu_tensor.gelu();
        let gpu_slice = gpu_result.as_slice().await.unwrap();

        // CPU reference
        let cpu_tensor: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([8], &data));
        let cpu_result = cpu_tensor.gelu();
        let cpu_slice = cpu_result.as_slice().await.unwrap();

        for i in 0..data.len() {
            eprintln!(
                "gelu({:>8.1}) : gpu={:>12.4}, cpu={:>12.4}, diff={:.6}",
                data[i],
                gpu_slice[[i]],
                cpu_slice[[i]],
                (gpu_slice[[i]] - cpu_slice[[i]]).abs()
            );
        }

        // Also test individual GPU ops to find where the bug is
        eprintln!("\n--- Intermediate GPU ops ---");

        // Test tanh on large values
        let tanh_data = [10.0f32, 100.0, 1000.0, 1e6, 1e8];
        let tanh_tensor: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [5], &tanh_data);
        let tanh_result = tanh_tensor.tanh();
        let tanh_slice = tanh_result.as_slice().await.unwrap();
        for (i, &x) in tanh_data.iter().enumerate() {
            eprintln!("GPU tanh({:>10.1}) = {}", x, tanh_slice.as_slice()[i]);
        }

        // Test x * x for large values
        let sq_data = [10.0f32, 100.0, 500.0, 1000.0, 2725.0];
        let sq_tensor: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [5], &sq_data);
        let sq_result = (&sq_tensor * &sq_tensor).to_concrete();
        let sq_slice = sq_result.clone().as_slice().await.unwrap();
        for (i, &x) in sq_data.iter().enumerate() {
            eprintln!(
                "GPU {}^2 = {} (expected {})",
                x,
                sq_slice.as_slice()[i],
                x * x
            );
        }

        // Test the inner factor: 0.044715 * x^2 + 1
        let inner_factor = (sq_result * 0.044715f32 + 1.0f32).to_concrete();
        let inner_slice = inner_factor.as_slice().await.unwrap();
        for (i, &x) in sq_data.iter().enumerate() {
            eprintln!(
                "GPU inner_factor({}) = {} (expected {})",
                x,
                inner_slice.as_slice()[i],
                0.044715 * x * x + 1.0
            );
        }

        // For large positive x, gelu(x) ≈ x
        assert!(
            (gpu_slice[[4]] - 2725.0).abs() < 1.0,
            "GPU gelu(2725) should be ~2725, got {}",
            gpu_slice[[4]]
        );
    }

    #[tokio::test]
    async fn test_gelu_cpu_vs_gpu() {
        use crate::Device;

        // Create random-ish data similar to FFN activations
        let data: Vec<f32> = (0..1 * 100 * 1536)
            .map(|i| (i as f32 * 0.001).sin() * 5.0)
            .collect();

        // CPU version
        let cpu_tensor: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 100, 1536], &data));
        let cpu_result = cpu_tensor.gelu();
        let cpu_slice = cpu_result.as_slice().await.unwrap();

        // GPU version
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let gpu_tensor: Tensor<3, f32> = Tensor::from_slice(&gpu_device, [1, 100, 1536], &data);
        let gpu_result = gpu_tensor.gelu();
        let gpu_slice = gpu_result.as_slice().await.unwrap();

        // Compare
        assert_eq!(cpu_slice.shape(), gpu_slice.shape());

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        let mut count = 0;
        for i in 0..cpu_slice.shape()[0] {
            for j in 0..cpu_slice.shape()[1].min(50) {
                for k in 0..cpu_slice.shape()[2].min(100) {
                    let cpu_val: f32 = cpu_slice[[i, j, k]].into();
                    let gpu_val: f32 = gpu_slice[[i, j, k]].into();
                    let diff = (cpu_val - gpu_val).abs();
                    max_diff = max_diff.max(diff);
                    sum_diff += diff;
                    count += 1;
                }
            }
        }

        eprintln!(
            "GELU CPU vs GPU: max_diff={}, mean_diff={}",
            max_diff,
            sum_diff / count as f32
        );
        eprintln!(
            "CPU[0,0,0..5]: {:?}",
            (0..5).map(|i| cpu_slice[[0, 0, i]]).collect::<Vec<f32>>()
        );
        eprintln!(
            "GPU[0,0,0..5]: {:?}",
            (0..5).map(|i| gpu_slice[[0, 0, i]]).collect::<Vec<f32>>()
        );

        assert!(
            max_diff < 0.01,
            "GELU CPU and GPU outputs differ too much: max_diff={}",
            max_diff
        );
    }
}
