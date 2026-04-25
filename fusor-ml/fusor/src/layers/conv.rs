//! N-dimensional convolution layer.

use crate::{ConcreteTensor, Device, MatmulImpl, SimdElement, Tensor, VarBuilder};
use fusor_core::{DataType, FloatDataType, LargerRank};
use fusor_cpu::{FloatOps, TensorBacking};

/// Configuration for an N-D convolution.
#[derive(Debug, Clone, Copy)]
pub struct ConvNdConfig<const SPATIAL: usize> {
    pub padding: [usize; SPATIAL],
    pub stride: [usize; SPATIAL],
    pub groups: usize,
}

impl<const SPATIAL: usize> Default for ConvNdConfig<SPATIAL> {
    fn default() -> Self {
        Self {
            padding: [0; SPATIAL],
            stride: [1; SPATIAL],
            groups: 1,
        }
    }
}

/// N-dimensional convolution layer.
///
/// Input / output tensors have rank `RANK = SPATIAL + 2`:
/// `(batch, channels, ...spatial)`.
/// Weight has shape `(out_channels, in_channels / groups, ...kernel)`.
pub struct ConvNd<const SPATIAL: usize, const RANK: usize, D: SimdElement> {
    weight: Tensor<RANK, D, ConcreteTensor<D, RANK>>,
    bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
    config: ConvNdConfig<SPATIAL>,
    in_channels: usize,
    out_channels: usize,
}

impl<const SPATIAL: usize, const RANK: usize, D> ConvNd<SPATIAL, RANK, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Create a new convolution layer.
    ///
    /// `weight` shape: `(out_channels, in_channels / groups, ...kernel)`.
    /// `bias` shape: `(out_channels,)`.
    pub fn new(
        weight: Tensor<RANK, D, ConcreteTensor<D, RANK>>,
        bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
        config: ConvNdConfig<SPATIAL>,
    ) -> Self {
        // RANK = SPATIAL + 2 (batch + channels + spatial). Compile-time so a
        // misuse like `ConvNd::<2, 5, _>::new(...)` is rejected at the type
        // level rather than panicking at runtime.
        const {
            assert!(RANK == SPATIAL + 2);
        }
        let shape = weight.shape();
        let out_channels = shape[0];
        let in_channels = shape[1] * config.groups;

        assert!(config.groups >= 1, "groups must be >= 1");
        assert_eq!(
            out_channels % config.groups,
            0,
            "out_channels ({out_channels}) must be divisible by groups ({})",
            config.groups
        );

        if let Some(ref b) = bias {
            assert_eq!(
                b.shape()[0],
                out_channels,
                "Bias shape must match out_channels"
            );
        }

        Self {
            weight,
            bias,
            config,
            in_channels,
            out_channels,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &ConvNdConfig<SPATIAL> {
        &self.config
    }

    /// Number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

impl<const SPATIAL: usize, const RANK: usize, D> ConvNd<SPATIAL, RANK, D>
where
    D: SimdElement
        + DataType
        + FloatDataType
        + FloatOps
        + Default
        + MatmulImpl
        + std::ops::Mul<Output = D>
        + std::ops::Add<Output = D>,
{
    /// Forward pass for any spatial rank. The free const generic `R2` equals
    /// `RANK + SPATIAL` and is determined by the `LargerRank` bound, exactly
    /// the same way the underlying `composite::conv` operation infers it.
    pub fn forward<B, const R2: usize>(
        &self,
        input: &Tensor<RANK, D, B>,
    ) -> Tensor<RANK, D, ConcreteTensor<D, RANK>>
    where
        B: TensorBacking<RANK, Elem = D>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
        ConcreteTensor<D, RANK>: fusor_cpu::LargerRank<R2, SPATIAL, D>,
        fusor_core::Tensor<RANK, D>: LargerRank<SPATIAL, R2, D>,
    {
        let input = input.to_concrete();

        if self.config.groups == 1 {
            return input.conv(
                &self.weight,
                self.bias.as_ref(),
                self.config.padding,
                self.config.stride,
            );
        }

        let g = self.config.groups;
        let in_ch_per_group = self.in_channels / g;
        let out_ch_per_group = self.out_channels / g;
        let mut group_outputs = Vec::with_capacity(g);
        for i in 0..g {
            let input_slice: Tensor<RANK, D, ConcreteTensor<D, RANK>> = input
                .narrow(1, i * in_ch_per_group, in_ch_per_group)
                .to_concrete();
            let weight_slice: Tensor<RANK, D, ConcreteTensor<D, RANK>> = self
                .weight
                .narrow(0, i * out_ch_per_group, out_ch_per_group)
                .to_concrete();
            let group_out: Tensor<RANK, D, ConcreteTensor<D, RANK>> = input_slice.conv(
                &weight_slice,
                None::<&Tensor<1, D, ConcreteTensor<D, 1>>>,
                self.config.padding,
                self.config.stride,
            );
            group_outputs.push(group_out);
        }
        let cat = Tensor::cat(group_outputs, 1);
        if let Some(bias) = &self.bias {
            let out_shape = cat.shape();
            let mut bias_shape = [1usize; RANK];
            bias_shape[1] = self.out_channels;
            let bias_reshaped = bias.reshape(bias_shape);
            let bias_nd: Tensor<RANK, D, _> = bias_reshaped.broadcast_as(out_shape);
            (cat + bias_nd).to_concrete()
        } else {
            cat.to_concrete()
        }
    }
}

impl<const SPATIAL: usize, const RANK: usize> ConvNd<SPATIAL, RANK, f32> {
    /// Load from `VarBuilder`; bias is optional and loaded if present.
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: ConvNdConfig<SPATIAL>,
    ) -> crate::Result<Self> {
        let weight: Tensor<RANK, f32> = vb.get("weight", device)?.dequantize();
        let bias: Option<Tensor<1, f32, ConcreteTensor<f32, 1>>> =
            vb.get("bias", device).ok().map(|b| b.dequantize());

        Ok(Self::new(weight.to_concrete(), bias, config))
    }

    /// Load from `VarBuilder` without a bias term.
    pub fn load_no_bias(
        device: &Device,
        vb: &mut VarBuilder,
        config: ConvNdConfig<SPATIAL>,
    ) -> crate::Result<Self> {
        let weight: Tensor<RANK, f32> = vb.get("weight", device)?.dequantize();
        Ok(Self::new(weight.to_concrete(), None, config))
    }
}

#[cfg(test)]
#[allow(clippy::useless_conversion)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conv1d_simple() {
        let weight_data = [0.2f32, 0.5, 0.3];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &weight_data));

        let bias_data = [0.1f32];
        let bias: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1], &bias_data));

        let conv = ConvNd::<1, 3, _>::new(weight, Some(bias), ConvNdConfig::<1>::default());

        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 5], &input_data));

        let output = conv.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 1, 3]);
        assert!((result[[0, 0, 0]] - 2.2).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 3.2).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 4.2).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv1d_with_padding() {
        let weight_data = [1.0f32, 1.0, 1.0];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &weight_data));

        let config = ConvNdConfig::<1> {
            padding: [1],
            ..Default::default()
        };
        let conv = ConvNd::<1, 3, _>::new(weight, None, config);

        let input_data = [1.0f32, 2.0, 3.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &input_data));

        let output = conv.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 1, 3]);
        assert!((result[[0, 0, 0]] - 3.0).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 6.0).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv1d_properties() {
        let weight_data = [0.0f32; 6];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3, 1], &weight_data));

        let config = ConvNdConfig::<1> {
            padding: [2],
            stride: [3],
            groups: 1,
        };
        let conv = ConvNd::<1, 3, _>::new(weight, None, config);

        assert_eq!(conv.in_channels(), 3);
        assert_eq!(conv.out_channels(), 2);
        assert_eq!(conv.config().padding, [2]);
        assert_eq!(conv.config().stride, [3]);
    }

    #[tokio::test]
    async fn test_conv1d_cpu_vs_gpu() {
        let weight_data: Vec<f32> = (0..384 * 80 * 3)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        let bias_data: Vec<f32> = (0..384).map(|i| (i as f32 * 0.0001).cos() * 0.01).collect();
        let input_data: Vec<f32> = (0..80 * 3000).map(|i| (i as f32 * 0.0001).sin()).collect();

        let config = ConvNdConfig::<1> {
            padding: [1],
            stride: [1],
            groups: 1,
        };

        let weight_cpu: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([384, 80, 3], &weight_data));
        let bias_cpu: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([384], &bias_data));
        let input_cpu: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 80, 3000], &input_data));
        let conv_cpu = ConvNd::<1, 3, _>::new(weight_cpu, Some(bias_cpu), config);
        let output_cpu = conv_cpu.forward(&input_cpu);
        let result_cpu = output_cpu.as_slice().await.unwrap();

        let gpu_device = Device::new().await.expect("GPU required for this test");
        let weight_gpu: Tensor<3, f32> =
            Tensor::from_slice(&gpu_device, [384, 80, 3], &weight_data);
        let bias_gpu: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [384], &bias_data);
        let input_gpu: Tensor<3, f32> = Tensor::from_slice(&gpu_device, [1, 80, 3000], &input_data);
        let conv_gpu = ConvNd::<1, 3, _>::new(weight_gpu, Some(bias_gpu), config);
        let output_gpu = conv_gpu.forward(&input_gpu);
        let result_gpu = output_gpu.as_slice().await.unwrap();

        assert_eq!(result_cpu.shape(), result_gpu.shape());

        let mut max_diff = 0.0f32;
        for i in 0..result_cpu.shape()[0] {
            for j in 0..result_cpu.shape()[1] {
                for k in 0..result_cpu.shape()[2].min(100) {
                    let cpu_val: f32 = result_cpu[[i, j, k]].into();
                    let gpu_val: f32 = result_gpu[[i, j, k]].into();
                    max_diff = max_diff.max((cpu_val - gpu_val).abs());
                }
            }
        }

        assert!(
            max_diff < 0.01,
            "Conv1d CPU and GPU outputs differ too much: max_diff={max_diff}"
        );
    }
}
