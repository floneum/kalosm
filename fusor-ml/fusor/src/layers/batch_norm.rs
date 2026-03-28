//! BatchNorm1d layer implementation for inference.

use crate::{DataType, Device, SimdElement, Tensor, VarBuilder};
use fusor_core::FloatDataType;
use fusor_cpu::{FloatOps, TensorBacking};

/// Inference-only BatchNorm1d.
///
/// Input shape: (batch, channels, length)
pub struct BatchNorm1d<D: SimdElement> {
    weight: Option<Tensor<1, D>>,
    bias: Option<Tensor<1, D>>,
    running_mean: Tensor<1, D>,
    running_var: Tensor<1, D>,
    eps: D,
}

impl<D> BatchNorm1d<D>
where
    D: SimdElement
        + DataType
        + FloatDataType
        + FloatOps
        + Default
        + std::ops::Add<Output = D>
        + std::ops::Sub<Output = D>
        + std::ops::Mul<Output = D>
        + std::ops::Div<Output = D>,
    crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
    crate::SubOp: fusor_cpu::SimdBinaryOp<D>,
    crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
    crate::DivOp: fusor_cpu::SimdBinaryOp<D>,
    crate::SqrtOp: fusor_cpu::SimdUnaryOp<D>,
{
    /// Create a new BatchNorm1d layer.
    pub fn new(
        weight: Option<Tensor<1, D>>,
        bias: Option<Tensor<1, D>>,
        running_mean: Tensor<1, D>,
        running_var: Tensor<1, D>,
        eps: D,
    ) -> Self {
        Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps,
        }
    }

    /// Forward pass.
    pub fn forward<B>(&self, input: &Tensor<3, D, B>) -> Tensor<3, D>
    where
        B: TensorBacking<3, Elem = D>,
    {
        let shape = input.shape();
        let channels = shape[1];
        assert_eq!(
            self.running_mean.shape()[0],
            channels,
            "running_mean channels ({}) must match input channels ({channels})",
            self.running_mean.shape()[0]
        );
        assert_eq!(
            self.running_var.shape()[0],
            channels,
            "running_var channels ({}) must match input channels ({channels})",
            self.running_var.shape()[0]
        );

        let mean_reshaped = self.running_mean.reshape([1, channels, 1]);
        let mean = mean_reshaped.broadcast_as(shape);
        let var_reshaped = self.running_var.reshape([1, channels, 1]);
        let var = var_reshaped.broadcast_as(shape);
        let normalized = (input.to_concrete() - mean)
            .to_concrete()
            .div_(&(var.to_concrete().add_scalar(self.eps).sqrt().broadcast_as(shape)));

        let scaled = if let Some(weight) = &self.weight {
            let weight_reshaped = weight.reshape([1, channels, 1]);
            let weight = weight_reshaped.broadcast_as(shape);
            normalized.mul_(&weight)
        } else {
            normalized
        };

        if let Some(bias) = &self.bias {
            let bias_reshaped = bias.reshape([1, channels, 1]);
            let bias = bias_reshaped.broadcast_as(shape);
            scaled.add_(&bias)
        } else {
            scaled
        }
    }
}

impl BatchNorm1d<f32> {
    /// Load a BatchNorm1d layer from a GGUF var builder.
    pub fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> crate::Result<Self> {
        let weight = vb.get("weight", device).ok().map(|w| w.dequantize());
        let bias = vb.get("bias", device).ok().map(|b| b.dequantize());
        let running_mean = vb.get("running_mean", device)?.dequantize();
        let running_var = vb.get("running_var", device)?.dequantize();
        Ok(Self::new(weight, bias, running_mean, running_var, eps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_norm_1d_inference() {
        let device = Device::Cpu;
        let input: Tensor<3, f32> =
            Tensor::from_slice(&device, [1, 2, 2], &[1.0, 3.0, 10.0, 14.0]);
        let weight: Tensor<1, f32> = Tensor::from_slice(&device, [2], &[2.0, 0.5]);
        let bias: Tensor<1, f32> = Tensor::from_slice(&device, [2], &[1.0, -1.0]);
        let mean: Tensor<1, f32> = Tensor::from_slice(&device, [2], &[2.0, 12.0]);
        let var: Tensor<1, f32> = Tensor::from_slice(&device, [2], &[4.0, 16.0]);

        let bn = BatchNorm1d::new(Some(weight), Some(bias), mean, var, 0.0);
        let output = bn.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert!((result[[0, 0, 0]] - 0.0).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 2.0).abs() < 1e-5);
        assert!((result[[0, 1, 0]] - -1.25).abs() < 1e-5);
        assert!((result[[0, 1, 1]] - -0.75).abs() < 1e-5);
    }
}
