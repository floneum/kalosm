//! LayerNorm2d implementation for normalizing over the channel dimension of BCHW tensors.

use crate::{ConcreteTensor, Device, SimdElement, Tensor, VarBuilder};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::{FloatOps, TensorBacking};

/// Layer Normalization for 2D spatial data (channel-wise normalization).
///
/// Unlike standard `LayerNorm` which normalizes the last dimension,
/// `LayerNorm2d` normalizes over the channel dimension (dim=1) of BCHW tensors.
///
/// Formula: `output = (input - mean) / sqrt(variance + eps) * weight + bias`
/// where mean and variance are computed over the channel dimension.
pub struct LayerNorm2d<D: SimdElement = f32> {
    weight: Tensor<1, D, ConcreteTensor<D, 1>>,
    bias: Tensor<1, D, ConcreteTensor<D, 1>>,
    num_channels: usize,
    eps: f32,
}

impl<D> LayerNorm2d<D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default + fusor_cpu::Scalar,
{
    /// Create a new LayerNorm2d layer.
    pub fn new(
        weight: Tensor<1, D, ConcreteTensor<D, 1>>,
        bias: Tensor<1, D, ConcreteTensor<D, 1>>,
        num_channels: usize,
        eps: f32,
    ) -> Self {
        Self {
            weight,
            bias,
            num_channels,
            eps,
        }
    }

    /// Forward pass for 4D input (batch, channels, height, width).
    ///
    /// Normalizes over the channel dimension (dim=1).
    pub fn forward<B>(&self, xs: &Tensor<4, D, B>) -> Tensor<4, D>
    where
        B: TensorBacking<4, Elem = D>,
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
    {
        let shape = xs.shape();

        let u: Tensor<4, D> = xs.mean_keepdim(1);
        let u_broadcast: Tensor<4, D> = u.broadcast_as(shape).to_concrete();
        let xs_centered: Tensor<4, D> = (xs - &u_broadcast).to_concrete();
        let s: Tensor<4, D> = (&xs_centered * &xs_centered).to_concrete().mean_keepdim(1);
        let s_eps = (s + D::from_f32(self.eps)).to_concrete();
        let denom: Tensor<4, D> = s_eps.sqrt().broadcast_as(shape).to_concrete();
        let xs_norm: Tensor<4, D> = (&xs_centered / denom).to_concrete();

        let w: Tensor<4, D> = self
            .weight
            .reshape([1, self.num_channels, 1, 1])
            .broadcast_as(shape)
            .to_concrete();
        let b: Tensor<4, D> = self
            .bias
            .reshape([1, self.num_channels, 1, 1])
            .broadcast_as(shape)
            .to_concrete();

        (xs_norm * w).to_concrete().add_(&b)
    }
}

impl LayerNorm2d<f32> {
    /// Load LayerNorm2d from VarBuilder.
    pub fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> crate::Result<Self> {
        let weight: Tensor<1, f32> = vb.get("weight", device)?.dequantize();
        let bias: Tensor<1, f32> = vb.get("bias", device)?.dequantize();
        let num_channels = weight.shape()[0];
        Ok(Self::new(
            weight.to_concrete(),
            bias.to_concrete(),
            num_channels,
            eps,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU sanity: with weight=1, bias=0 the output of each (b, :, h, w) column
    /// should have mean 0 and unit variance.
    #[tokio::test]
    async fn test_layer_norm_2d_cpu() {
        let device = Device::Cpu;
        // Channels = 4
        let weight: Tensor<1, f32> = Tensor::from_slice(&device, [4], &[1.0; 4]);
        let bias: Tensor<1, f32> = Tensor::from_slice(&device, [4], &[0.0; 4]);
        let ln = LayerNorm2d::new(weight.to_concrete(), bias.to_concrete(), 4, 1e-5);

        // (1, 4, 1, 1) — single spatial column with values [1, 2, 3, 4]
        let xs: Tensor<4, f32> = Tensor::from_slice(&device, [1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0]);
        let out = ln.forward(&xs);
        let slice = out.as_slice().await.unwrap();

        // Mean of [1,2,3,4] is 2.5. Variance is 1.25, std ≈ 1.118.
        // Normalized values: [-1.341, -0.447, 0.447, 1.341]
        let expected = [-1.3416, -0.4472, 0.4472, 1.3416];
        for c in 0..4 {
            assert!(
                (slice[[0, c, 0, 0]] - expected[c]).abs() < 1e-3,
                "channel {} got {}, want {}",
                c,
                slice[[0, c, 0, 0]],
                expected[c]
            );
        }
    }

    #[cfg(not(target_os = "windows"))]
    #[tokio::test]
    async fn test_layer_norm_2d_cpu_vs_gpu() {
        let cpu = Device::Cpu;
        let gpu = Device::new().await.expect("GPU required for this test");

        let channels = 8usize;
        let h = 3usize;
        let w = 5usize;
        let weight_data: Vec<f32> = (0..channels).map(|i| 1.0 + (i as f32) * 0.1).collect();
        let bias_data: Vec<f32> = (0..channels).map(|i| (i as f32) * 0.05).collect();
        let xs_data: Vec<f32> = (0..channels * h * w)
            .map(|i| ((i as f32) * 0.07).sin())
            .collect();

        let make = |dev: &Device| -> LayerNorm2d<f32> {
            let weight: Tensor<1, f32> = Tensor::from_slice(dev, [channels], &weight_data);
            let bias: Tensor<1, f32> = Tensor::from_slice(dev, [channels], &bias_data);
            LayerNorm2d::new(weight.to_concrete(), bias.to_concrete(), channels, 1e-5)
        };
        let cpu_ln = make(&cpu);
        let gpu_ln = make(&gpu);

        let cpu_xs: Tensor<4, f32> = Tensor::from_slice(&cpu, [1, channels, h, w], &xs_data);
        let gpu_xs: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, channels, h, w], &xs_data);
        let cpu_slice = cpu_ln.forward(&cpu_xs).as_slice().await.unwrap();
        let gpu_slice = gpu_ln.forward(&gpu_xs).as_slice().await.unwrap();
        assert_eq!(cpu_slice.shape(), gpu_slice.shape());

        let mut max_diff = 0.0f32;
        for c in 0..channels {
            for r in 0..h {
                for col in 0..w {
                    let a: f32 = cpu_slice[[0, c, r, col]].into();
                    let b: f32 = gpu_slice[[0, c, r, col]].into();
                    max_diff = max_diff.max((a - b).abs());
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "LayerNorm2d CPU vs GPU diverged: max_diff={}",
            max_diff
        );
    }
}
