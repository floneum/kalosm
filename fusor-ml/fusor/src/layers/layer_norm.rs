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
            Tensor::Gpu(_) => {
                // Fall back to standard forward for GPU
                self.forward(input)
            }
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
}
