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
    pub fn forward_fused<B>(&self, input: &Tensor<3, f32, B>) -> Tensor<3, f32>
    where
        B: TensorBacking<3, Elem = f32>,
    {
        input.layer_norm_last_dim_fused::<2, 1, _, _>(&self.weight, self.bias.as_ref(), self.eps)
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

