//! N-dimensional layer normalization.

use crate::{ConcreteTensor, Device, SimdElement, Tensor, VarBuilder};
use fusor_core::{DataType, FloatDataType, LastRank as GpuLastRank, LastRankInner, NextRankInner};
use fusor_cpu::{FloatOps, LastRank as CpuLastRank, TensorBacking};

/// Layer normalization with a selectable reduction axis.
///
/// `axis == None` normalizes the last dimension (standard transformer
/// LayerNorm). `axis == Some(a)` normalizes dimension `a` (used for
/// channel-wise normalization on BCHW tensors with `axis = 1`).
pub struct LayerNormNd<D: SimdElement = f32> {
    weight: Tensor<1, D, ConcreteTensor<D, 1>>,
    bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
    axis: Option<usize>,
    eps: f32,
}

impl<D> LayerNormNd<D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Create a LayerNorm that normalizes the last dimension.
    pub fn new(
        weight: Tensor<1, D, ConcreteTensor<D, 1>>,
        bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
        eps: f32,
    ) -> Self {
        Self {
            weight,
            bias,
            axis: None,
            eps,
        }
    }

    /// Create a LayerNorm that normalizes the given axis.
    pub fn new_over_axis(
        weight: Tensor<1, D, ConcreteTensor<D, 1>>,
        bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
        axis: usize,
        eps: f32,
    ) -> Self {
        Self {
            weight,
            bias,
            axis: Some(axis),
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor<1, D, ConcreteTensor<D, 1>> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<1, D, ConcreteTensor<D, 1>>> {
        self.bias.as_ref()
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Forward pass for any input rank. `OUT_RANK` equals `N - 1` and is
    /// determined by the `LastRank` bound — same inference mechanism the
    /// `layer_norm` / `mean_keepdim` ops use.
    pub fn forward<const N: usize, const OUT_RANK: usize, B>(
        &self,
        input: &Tensor<N, D, B>,
    ) -> Tensor<N, D, ConcreteTensor<D, N>>
    where
        B: TensorBacking<N, Elem = D>,
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
        ConcreteTensor<D, N>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<N, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<N, D> as LastRankInner>::LastRank:
            NextRankInner<NextRank = fusor_core::Tensor<N, D>>,
    {
        let shape = input.shape();
        let axis = self.axis.unwrap_or(N - 1);

        if axis == N - 1 {
            let weight_b: Tensor<N, D, _> = self.weight.broadcast_as(shape);
            let bias_b: Option<Tensor<N, D, _>> =
                self.bias.as_ref().map(|b| b.broadcast_as(shape));
            return input.layer_norm(&weight_b, bias_b.as_ref(), D::from_f32(self.eps), true);
        }

        let num_features = shape[axis];

        let mean: Tensor<N, D> = input.mean_keepdim::<OUT_RANK>(axis);
        let mean_b: Tensor<N, D> = mean.broadcast_as(shape).to_concrete();
        let centered: Tensor<N, D> = (input - &mean_b).to_concrete();
        let var: Tensor<N, D> = (&centered * &centered)
            .to_concrete()
            .mean_keepdim::<OUT_RANK>(axis);
        let var_eps = var.to_concrete().add_scalar(D::from_f32(self.eps));
        let denom: Tensor<N, D> = var_eps.sqrt().broadcast_as(shape).to_concrete();
        let normed: Tensor<N, D> = (&centered / denom).to_concrete();

        let mut affine_shape = [1usize; N];
        affine_shape[axis] = num_features;
        let w: Tensor<N, D> = self
            .weight
            .reshape(affine_shape)
            .broadcast_as(shape)
            .to_concrete();
        let scaled = (normed * w).to_concrete();
        if let Some(bias) = &self.bias {
            let b: Tensor<N, D> = bias
                .reshape(affine_shape)
                .broadcast_as(shape)
                .to_concrete();
            scaled.add_(&b)
        } else {
            scaled
        }
    }
}

impl LayerNormNd<f32> {
    /// Load a last-dim LayerNorm from a `VarBuilder`. Bias is optional.
    pub fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> crate::Result<Self> {
        let weight_q = vb.get("weight", device)?;
        let weight_shape = weight_q.shape();

        let weight: Tensor<1, f32> = if weight_shape.len() == 1 {
            weight_q.dequantize()
        } else {
            let weight_2d: Tensor<2, f32> = weight_q.dequantize();
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

    /// Load a LayerNorm that normalizes `axis`. Bias is optional.
    pub fn load_over_axis(
        device: &Device,
        vb: &mut VarBuilder,
        axis: usize,
        eps: f32,
    ) -> crate::Result<Self> {
        let weight: Tensor<1, f32> = vb.get("weight", device)?.dequantize();
        let bias: Option<Tensor<1, f32, ConcreteTensor<f32, 1>>> =
            vb.get("bias", device).ok().map(|b| b.dequantize());
        Ok(Self::new_over_axis(weight.to_concrete(), bias, axis, eps))
    }

    /// Fused CPU fast path for normalizing the last dim of a rank-3 tensor.
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
            Tensor::Gpu(_) => self.forward(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_layer_norm_2d_input() {
        let weight_data = [1.0f32, 1.0, 1.0];
        let bias_data = [0.0f32, 0.0, 0.0];
        let weight: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &weight_data));
        let bias: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &bias_data));

        let layer_norm: LayerNormNd<f32> = LayerNormNd::new(weight, Some(bias), 1e-5);

        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &input_data));

        let output = layer_norm.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        let expected_val = (3.0f32 / 2.0).sqrt();
        assert!((result[[0, 0]] - (-expected_val)).abs() < 1e-4);
        assert!(result[[0, 1]].abs() < 1e-4);
        assert!((result[[0, 2]] - expected_val).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_layer_norm_3d_input() {
        let weight_data = [1.0f32, 1.0];
        let weight: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2], &weight_data));

        let layer_norm: LayerNormNd<f32> = LayerNormNd::new(weight, None, 1e-5);

        let input_data = [1.0f32, 3.0, 2.0, 4.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2], &input_data));

        let output = layer_norm.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2, 2]);
        assert!((result[[0, 0, 0]] - (-1.0)).abs() < 1e-4);
        assert!((result[[0, 0, 1]] - 1.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_layer_norm_channel_axis() {
        let device = Device::Cpu;
        let weight: Tensor<1, f32> = Tensor::from_slice(&device, [4], &[1.0; 4]);
        let bias: Tensor<1, f32> = Tensor::from_slice(&device, [4], &[0.0; 4]);
        let ln: LayerNormNd<f32> =
            LayerNormNd::new_over_axis(weight.to_concrete(), Some(bias.to_concrete()), 1, 1e-5);

        let xs: Tensor<4, f32> = Tensor::from_slice(&device, [1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0]);
        let out = ln.forward(&xs);
        let slice = out.as_slice().await.unwrap();

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
}
