//! N-dimensional layer normalization.

use crate::{ConcreteTensor, Device, SimdElement, Tensor, VarBuilder};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::{FloatOps, TensorBacking};

/// Layer normalization with a selectable reduction axis.
///
/// `axis == None` normalizes the last dimension (standard transformer
/// LayerNorm). `axis == Some(a)` normalizes dimension `a` (used for
/// channel-wise normalization on BCHW tensors with `axis = 1`).
pub struct LayerNormNd<const N: usize, D: SimdElement = f32> {
    weight: Tensor<1, D, ConcreteTensor<D, 1>>,
    bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
    axis: Option<usize>,
    eps: f32,
}

impl<const N: usize, D> LayerNormNd<N, D>
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
}

impl<D> LayerNormNd<2, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    pub fn forward<B>(&self, input: &Tensor<2, D, B>) -> Tensor<2, D, ConcreteTensor<D, 2>>
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
        debug_assert!(
            self.axis.is_none() || self.axis == Some(1),
            "LayerNormNd<2>: only last-dim normalization is supported"
        );
        let weight_b: Tensor<2, D, _> = self.weight.broadcast_as(input.shape());
        let bias_b: Option<Tensor<2, D, _>> =
            self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));
        input.layer_norm(&weight_b, bias_b.as_ref(), D::from_f32(self.eps), true)
    }
}

impl<D> LayerNormNd<3, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
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
        debug_assert!(
            self.axis.is_none() || self.axis == Some(2),
            "LayerNormNd<3>: only last-dim normalization is supported"
        );
        let weight_b: Tensor<3, D, _> = self.weight.broadcast_as(input.shape());
        let bias_b: Option<Tensor<3, D, _>> =
            self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));
        input.layer_norm(&weight_b, bias_b.as_ref(), D::from_f32(self.eps), true)
    }
}

impl<D> LayerNormNd<4, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default + fusor_cpu::Scalar,
{
    pub fn forward<B>(&self, input: &Tensor<4, D, B>) -> Tensor<4, D, ConcreteTensor<D, 4>>
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
        B: TensorBacking<4, Elem = D>,
    {
        let axis = self.axis.unwrap_or(3);
        if axis == 3 {
            let weight_b: Tensor<4, D, _> = self.weight.broadcast_as(input.shape());
            let bias_b: Option<Tensor<4, D, _>> =
                self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));
            return input.layer_norm(&weight_b, bias_b.as_ref(), D::from_f32(self.eps), true);
        }

        let shape = input.shape();
        let num_features = shape[axis];

        let mean: Tensor<4, D> = input.mean_keepdim(axis);
        let mean_b: Tensor<4, D> = mean.broadcast_as(shape).to_concrete();
        let centered: Tensor<4, D> = (input - &mean_b).to_concrete();
        let var: Tensor<4, D> = (&centered * &centered).to_concrete().mean_keepdim(axis);
        let var_eps = (var + D::from_f32(self.eps)).to_concrete();
        let denom: Tensor<4, D> = var_eps.sqrt().broadcast_as(shape).to_concrete();
        let normed: Tensor<4, D> = (&centered / denom).to_concrete();

        let mut affine_shape = [1usize; 4];
        affine_shape[axis] = num_features;
        let w: Tensor<4, D> = self
            .weight
            .reshape(affine_shape)
            .broadcast_as(shape)
            .to_concrete();
        let scaled = (normed * w).to_concrete();
        if let Some(bias) = &self.bias {
            let b: Tensor<4, D> = bias.reshape(affine_shape).broadcast_as(shape).to_concrete();
            scaled.add_(&b)
        } else {
            scaled
        }
    }
}

impl<const N: usize> LayerNormNd<N, f32> {
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
}

impl LayerNormNd<3, f32> {
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

        let layer_norm: LayerNormNd<2, f32> = LayerNormNd::new(weight, Some(bias), 1e-5);

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

        let layer_norm: LayerNormNd<3, f32> = LayerNormNd::new(weight, None, 1e-5);

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
        let ln: LayerNormNd<4, f32> = LayerNormNd::new_over_axis(
            weight.to_concrete(),
            Some(bias.to_concrete()),
            1,
            1e-5,
        );

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
