//! Layer normalization implementation.

use crate::{ConcreteTensor, Device, SimdElement, Tensor, VarBuilder};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::{FloatOps, TensorBacking};

/// Layer Normalization.
///
/// Normalizes the input over the last dimension.
/// Formula: output = (input - mean) / sqrt(variance + eps) * weight + bias
pub struct LayerNorm<const N: usize, D: SimdElement> {
    weight: Tensor<N, D, ConcreteTensor<D, N>>,
    bias: Option<Tensor<N, D, ConcreteTensor<D, N>>>,
    eps: f32,
}

impl<const N: usize, D> LayerNorm<N, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Create a new LayerNorm layer.
    ///
    /// Weight and bias should have shape (normalized_dim,).
    pub fn new(
        weight: Tensor<N, D, ConcreteTensor<D, N>>,
        bias: Option<Tensor<N, D, ConcreteTensor<D, N>>>,
        eps: f32,
    ) -> Self {
        Self { weight, bias, eps }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor<N, D, ConcreteTensor<D, N>> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor<N, D, ConcreteTensor<D, N>>> {
        self.bias.as_ref()
    }

    /// Get the epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl<D> LayerNorm<1, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Forward pass for 2D input (batch, features).
    ///
    /// Normalizes over the last dimension (features).
    pub fn forward_2d<B>(&self, input: &Tensor<2, D, B>) -> Tensor<2, D, ConcreteTensor<D, 2>>
    where
        D: std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        crate::SubOp: fusor_cpu::SimdBinaryOp<D>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::DivOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
        fusor_cpu::SqrtOp: fusor_cpu::SimdUnaryOp<D>,
        B: TensorBacking<2, Elem = D>,
    {
        // Broadcast weight to input shape
        let weight_broadcast: Tensor<2, D, _> = self.weight.broadcast_as(input.shape());
        let bias_broadcast: Option<Tensor<2, D, _>> =
            self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));
        input.layer_norm(
            &weight_broadcast,
            bias_broadcast.as_ref(),
            D::from_f32(self.eps),
            true,
        )
    }

    /// Forward pass for 3D input (batch, seq_len, features).
    ///
    /// Normalizes over the last dimension (features).
    pub fn forward<B>(&self, input: &Tensor<3, D, B>) -> Tensor<3, D, ConcreteTensor<D, 3>>
    where
        D: std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        crate::SubOp: fusor_cpu::SimdBinaryOp<D>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::DivOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
        fusor_cpu::SqrtOp: fusor_cpu::SimdUnaryOp<D>,
        B: TensorBacking<3, Elem = D>,
    {
        // Broadcast weight to input shape
        let weight_broadcast: Tensor<3, D, _> = self.weight.broadcast_as(input.shape());
        let bias_broadcast: Option<Tensor<3, D, _>> =
            self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));
        input.layer_norm(
            &weight_broadcast,
            bias_broadcast.as_ref(),
            D::from_f32(self.eps),
            true,
        )
    }
}

impl LayerNorm<1, f32> {
    /// Forward pass with fused CPU kernel (3D input).
    ///
    /// This is significantly faster than the standard forward pass on CPU
    /// as it computes mean, variance, and normalization in fewer passes.
    pub fn forward_fused<B>(
        &self,
        input: &Tensor<3, f32, B>,
    ) -> Tensor<3, f32, ConcreteTensor<f32, 3>>
    where
        B: TensorBacking<3, Elem = f32>,
    {
        match input {
            Tensor::Cpu(t) => {
                let contiguous = t.to_concrete();
                // Broadcast weight to match input shape for fused kernel
                let weight_broadcast = self.weight.broadcast_as(input.shape());
                let bias_broadcast = self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));

                let (weight_inner, bias_inner) = match (&weight_broadcast, &bias_broadcast) {
                    (Tensor::Cpu(w), Some(Tensor::Cpu(b))) => (
                        w.to_concrete().inner().clone(),
                        Some(b.to_concrete().inner().clone()),
                    ),
                    (Tensor::Cpu(w), None) => (w.to_concrete().inner().clone(), None),
                    _ => unreachable!(),
                };

                let result = fusor_cpu::layer_norm_last_dim_fused(
                    contiguous.inner(),
                    &weight_inner,
                    bias_inner.as_ref(),
                    self.eps,
                );
                Tensor::Cpu(fusor_cpu::Tensor::new(result))
            }
            Tensor::Gpu(input) => match &self.weight {
                Tensor::Gpu(weight) => {
                    let gpu_bias = self.bias.as_ref().map(|bias| match bias {
                        Tensor::Gpu(bias) => bias,
                        _ => panic!("LayerNorm bias must be on GPU when input is on GPU"),
                    });
                    Tensor::Gpu(input.layer_norm_fused(weight, gpu_bias, self.eps))
                }
                _ => panic!("LayerNorm weight must be on GPU when input is on GPU"),
            },
        }
    }

    /// Load LayerNorm from VarBuilder.
    ///
    /// Expects:
    /// - weight: Tensor with shape matching the normalized dimension
    /// - bias (optional): Tensor with same shape as weight
    pub fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> crate::Result<Self> {
        let weight_q = vb.get("weight", device)?;
        let weight_shape = weight_q.shape();

        // Handle both 1D and 2D weight formats
        let weight: Tensor<1, f32> = if weight_shape.len() == 1 {
            weight_q.dequantize()
        } else {
            let weight_2d: Tensor<2, f32> = weight_q.dequantize();
            // Squeeze to 1D
            if weight_2d.shape()[0] == 1 {
                weight_2d.squeeze(0).to_concrete()
            } else {
                weight_2d.squeeze(1).to_concrete()
            }
        };

        let bias = vb.get("bias", device).ok().map(|b| {
            let bias_shape = b.shape();
            if bias_shape.len() == 1 {
                b.dequantize()
            } else {
                let bias_2d: Tensor<2, f32> = b.dequantize();
                if bias_2d.shape()[0] == 1 {
                    bias_2d.squeeze(0).to_concrete()
                } else {
                    bias_2d.squeeze(1).to_concrete()
                }
            }
        });

        Ok(Self::new(weight.to_concrete(), bias, eps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_layer_norm_2d() {
        // Weight and bias: (3,)
        let weight_data = [1.0f32, 1.0, 1.0];
        let bias_data = [0.0f32, 0.0, 0.0];
        let weight: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &weight_data));
        let bias: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &bias_data));

        let layer_norm = LayerNorm::new(weight, Some(bias), 1e-5);

        // Input: (2, 3)
        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &input_data));

        let output = layer_norm.forward_2d(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[2, 3]);

        // Each row should have mean ~0 and std ~1 after normalization
        // For [1, 2, 3]: mean=2, std=sqrt(2/3)
        // Normalized: [-sqrt(3/2), 0, sqrt(3/2)] ≈ [-1.22, 0, 1.22]
        let expected_val = (3.0f32 / 2.0).sqrt();
        assert!((result[[0, 0]] - (-expected_val)).abs() < 1e-4);
        assert!(result[[0, 1]].abs() < 1e-4);
        assert!((result[[0, 2]] - expected_val).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_layer_norm_3d() {
        let weight_data = [1.0f32, 1.0];
        let weight: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2], &weight_data));

        let layer_norm = LayerNorm::new(weight, None, 1e-5);

        // Input: (1, 2, 2)
        let input_data = [1.0f32, 3.0, 2.0, 4.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2], &input_data));

        let output = layer_norm.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2, 2]);

        // First position [1, 3]: mean=2, std=1, normalized=[-1, 1]
        assert!((result[[0, 0, 0]] - (-1.0)).abs() < 1e-4);
        assert!((result[[0, 0, 1]] - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_layer_norm_forward_fused_gpu_matches_cpu() {
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let input_data = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 13.0, -4.0, -1.0, 0.5, 3.0,
        ];
        let weight_data = [1.0f32, 0.5, 2.0, -1.0];
        let bias_data = [0.0f32, 0.25, -0.5, 1.0];

        let cpu_input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 3, 4], &input_data));
        let gpu_input: Tensor<3, f32> = Tensor::from_slice(&gpu_device, [1, 3, 4], &input_data);

        let cpu_weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &weight_data));
        let gpu_weight: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [4], &weight_data);
        let cpu_bias: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &bias_data));
        let gpu_bias: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [4], &bias_data);

        let cpu_layer_norm = LayerNorm::new(cpu_weight, Some(cpu_bias), 1e-5);
        let gpu_layer_norm = LayerNorm::new(gpu_weight, Some(gpu_bias), 1e-5);

        let cpu_output = cpu_layer_norm.forward_fused(&cpu_input);
        let gpu_output = gpu_layer_norm.forward_fused(&gpu_input);
        let cpu_output = cpu_output.as_slice().await.unwrap();
        let gpu_output = gpu_output.as_slice().await.unwrap();

        for (cpu, gpu) in cpu_output.as_slice().iter().zip(gpu_output.as_slice()) {
            let diff = (cpu - gpu).abs();
            assert!(
                diff < 1e-4,
                "CPU/GPU fused LayerNorm mismatch: {cpu} vs {gpu}"
            );
        }
    }

    #[tokio::test]
    async fn test_layer_norm_forward_fused_gpu_matches_cpu_non_contiguous_no_bias() {
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let input_data = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 13.0, -4.0, -1.0, 0.5, 3.0, 8.0, 6.0, 4.0, 2.0,
            0.25, 0.5, 1.0, 2.0, 3.0, 1.5, 0.75, 0.25, 9.0, 10.0, 12.0, 15.0, -8.0, -4.0, -2.0,
            -1.0,
        ];
        let weight_data = [1.0f32, 1.0, 1.0, 1.0];

        let cpu_input = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 4, 4], &input_data));
        let cpu_input = cpu_input.narrow(1, 1, 2);
        let gpu_input_full: Tensor<3, f32> =
            Tensor::from_slice(&gpu_device, [2, 4, 4], &input_data);
        let gpu_input = gpu_input_full.narrow(1, 1, 2);

        let cpu_weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &weight_data));
        let gpu_weight: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [4], &weight_data);

        let cpu_layer_norm = LayerNorm::new(cpu_weight, None, 1e-5);
        let gpu_layer_norm = LayerNorm::new(gpu_weight, None, 1e-5);

        let cpu_output = cpu_layer_norm.forward_fused(&cpu_input);
        let gpu_output = gpu_layer_norm.forward_fused(&gpu_input);
        let cpu_output = cpu_output.as_slice().await.unwrap();
        let gpu_output = gpu_output.as_slice().await.unwrap();

        for (cpu, gpu) in cpu_output.as_slice().iter().zip(gpu_output.as_slice()) {
            let diff = (cpu - gpu).abs();
            assert!(
                diff < 1e-4,
                "CPU/GPU fused LayerNorm mismatch: {cpu} vs {gpu}"
            );
        }
    }

    #[tokio::test]
    async fn test_layer_norm_forward_fused_gpu_matches_cpu_hidden_320() {
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let hidden = 320;
        let input_data: Vec<f32> = (0..2 * 3 * hidden)
            .map(|i| (((i * 13) % 41) as f32 - 20.0) * 0.0625)
            .collect();
        let weight_data: Vec<f32> = (0..hidden)
            .map(|i| 0.85 + (i % 11) as f32 * 0.0125)
            .collect();

        let cpu_input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3, hidden], &input_data));
        let gpu_input: Tensor<3, f32> =
            Tensor::from_slice(&gpu_device, [2, 3, hidden], &input_data);
        let cpu_weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([hidden], &weight_data));
        let gpu_weight: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [hidden], &weight_data);

        let cpu_layer_norm = LayerNorm::new(cpu_weight, None, 1e-5);
        let gpu_layer_norm = LayerNorm::new(gpu_weight, None, 1e-5);

        let cpu_output = cpu_layer_norm.forward_fused(&cpu_input);
        let gpu_output = gpu_layer_norm.forward_fused(&gpu_input);
        let cpu_output = cpu_output.as_slice().await.unwrap();
        let gpu_output = gpu_output.as_slice().await.unwrap();

        for (cpu, gpu) in cpu_output.as_slice().iter().zip(gpu_output.as_slice()) {
            let diff = (cpu - gpu).abs();
            assert!(
                diff < 1e-4,
                "CPU/GPU fused LayerNorm mismatch: {cpu} vs {gpu}"
            );
        }
    }

    #[tokio::test]
    async fn test_layer_norm_forward_fused_gpu_unit_offset_pattern() {
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let hidden = 320;
        let input_data: Vec<f32> = (0..2 * 3 * hidden)
            .map(|i| (((i * 17) % 53) as f32 - 26.0) * 0.03125)
            .collect();
        let ones = vec![1.0f32; hidden];
        let gamma_data: Vec<f32> = (0..hidden)
            .map(|i| ((i * 7) % 29) as f32 * 0.01 - 0.14)
            .collect();

        let cpu_input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3, hidden], &input_data));
        let gpu_input: Tensor<3, f32> =
            Tensor::from_slice(&gpu_device, [2, 3, hidden], &input_data);
        let cpu_weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([hidden], &ones));
        let gpu_weight: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [hidden], &ones);
        let cpu_gamma: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([hidden], &gamma_data));
        let gpu_gamma: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [hidden], &gamma_data);

        let cpu_layer_norm = LayerNorm::new(cpu_weight, None, 1e-5);
        let gpu_layer_norm = LayerNorm::new(gpu_weight, None, 1e-5);

        let cpu_normed = cpu_layer_norm.forward_fused(&cpu_input);
        let cpu_gamma_offset = cpu_gamma.add_scalar(1.0);
        let cpu_gamma_broadcast = cpu_gamma_offset.broadcast_as(cpu_normed.shape());
        let cpu_output = cpu_normed.mul_(&cpu_gamma_broadcast).to_concrete();
        let gpu_normed = gpu_layer_norm.forward_fused(&gpu_input);
        let gpu_gamma_offset = gpu_gamma.add_scalar(1.0);
        let gpu_gamma_broadcast = gpu_gamma_offset.broadcast_as(gpu_normed.shape());
        let gpu_output = gpu_normed.mul_(&gpu_gamma_broadcast).to_concrete();

        let cpu_output = cpu_output.as_slice().await.unwrap();
        let gpu_output = gpu_output.as_slice().await.unwrap();

        for (cpu, gpu) in cpu_output.as_slice().iter().zip(gpu_output.as_slice()) {
            let diff = (cpu - gpu).abs();
            assert!(
                diff < 1e-4,
                "CPU/GPU unit-offset LayerNorm mismatch: {cpu} vs {gpu}"
            );
        }
    }

    #[tokio::test]
    async fn test_layer_norm_forward_fused_gpu_low_variance_hidden_320() {
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let hidden = 320;
        let input_data: Vec<f32> = (0..2 * 3 * hidden)
            .map(|i| 1.0 + (((i * 13) % 17) as f32 - 8.0) * 1e-4)
            .collect();
        let weight_data = vec![1.0f32; hidden];

        let cpu_input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3, hidden], &input_data));
        let gpu_input: Tensor<3, f32> =
            Tensor::from_slice(&gpu_device, [2, 3, hidden], &input_data);
        let cpu_weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([hidden], &weight_data));
        let gpu_weight: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [hidden], &weight_data);

        let cpu_layer_norm = LayerNorm::new(cpu_weight, None, 1e-5);
        let gpu_layer_norm = LayerNorm::new(gpu_weight, None, 1e-5);

        let cpu_output = cpu_layer_norm.forward_fused(&cpu_input);
        let gpu_output = gpu_layer_norm.forward_fused(&gpu_input);
        let cpu_output = cpu_output.as_slice().await.unwrap();
        let gpu_output = gpu_output.as_slice().await.unwrap();

        for (cpu, gpu) in cpu_output.as_slice().iter().zip(gpu_output.as_slice()) {
            let diff = (cpu - gpu).abs();
            assert!(
                diff < 1e-4,
                "CPU/GPU low-variance fused LayerNorm mismatch: {cpu} vs {gpu}"
            );
        }
    }
}
