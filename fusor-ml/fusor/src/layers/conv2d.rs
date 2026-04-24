//! Conv2d layer implementation.

use crate::{ConcreteTensor, MatmulImpl, SimdElement, Tensor};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::FloatOps;

/// Configuration for Conv2d layers.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    pub padding: [usize; 2],
    pub stride: [usize; 2],
    pub groups: usize,
    pub dilation: [usize; 2],
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: [0, 0],
            stride: [1, 1],
            groups: 1,
            dilation: [1, 1],
        }
    }
}

/// 2D Convolution layer.
pub struct Conv2d<D: SimdElement> {
    weight: Tensor<4, D, ConcreteTensor<D, 4>>, // (out_channels, in_channels/groups, kernel_h, kernel_w)
    bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>, // (out_channels,)
    config: Conv2dConfig,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
}

impl<D> Conv2d<D>
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
    /// Create a new Conv2d layer with given weights and configuration.
    pub fn new(
        weight: Tensor<4, D, ConcreteTensor<D, 4>>,
        bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
        config: Conv2dConfig,
    ) -> Self {
        let shape = weight.shape();
        let out_channels = shape[0];
        let in_channels_per_group = shape[1];
        let kernel_size = [shape[2], shape[3]];

        assert!(
            config.groups > 0,
            "groups must be greater than zero, got {}",
            config.groups
        );
        assert_eq!(
            config.dilation,
            [1, 1],
            "Only dilation=[1, 1] is currently supported"
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
            in_channels: in_channels_per_group * config.groups,
            out_channels,
            kernel_size,
        }
    }

    /// Forward pass.
    pub fn forward(
        &self,
        input: &Tensor<4, D, ConcreteTensor<D, 4>>,
    ) -> Tensor<4, D, ConcreteTensor<D, 4>>
    where
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
    {
        if self.config.groups == 1 {
            return input.conv(
                &self.weight,
                self.bias.as_ref(),
                self.config.padding,
                self.config.stride,
            );
        }

        let in_channels_per_group = self.in_channels / self.config.groups;
        let out_channels_per_group = self.out_channels / self.config.groups;
        let mut outputs = Vec::with_capacity(self.config.groups);

        for group in 0..self.config.groups {
            let input_group = input
                .narrow(1, group * in_channels_per_group, in_channels_per_group)
                .to_concrete();
            let weight_group = self
                .weight
                .narrow(0, group * out_channels_per_group, out_channels_per_group)
                .to_concrete();
            let bias_group = self.bias.as_ref().map(|bias| {
                bias.narrow(0, group * out_channels_per_group, out_channels_per_group)
                    .to_concrete()
            });

            outputs.push(input_group.conv(
                &weight_group,
                bias_group.as_ref(),
                self.config.padding,
                self.config.stride,
            ));
        }

        Tensor::cat(outputs, 1)
    }

    /// Get the configuration.
    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }

    /// Get the number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get the number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> [usize; 2] {
        self.kernel_size
    }
}

impl Conv2d<f32> {
    /// Load a Conv2d layer from a GGUF var builder.
    pub fn load(
        device: &crate::Device,
        vb: &mut crate::VarBuilder,
        config: Conv2dConfig,
    ) -> crate::Result<Self> {
        let weight = vb.get("weight", device)?.dequantize();
        let bias = vb.get("bias", device).ok().map(|b| b.dequantize());
        Ok(Self::new(weight, bias, config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conv2d_simple() {
        let input: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, 1, 3, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ));
        let weight: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, 1, 2, 2],
            &[1.0, 0.0, 0.0, 1.0],
        ));

        let conv = Conv2d::new(weight, None, Conv2dConfig::default());
        let output = conv.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        assert!((result[[0, 0, 0, 0]] - 6.0).abs() < 1e-5);
        assert!((result[[0, 0, 0, 1]] - 8.0).abs() < 1e-5);
        assert!((result[[0, 0, 1, 0]] - 12.0).abs() < 1e-5);
        assert!((result[[0, 0, 1, 1]] - 14.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv2d_depthwise_groups() {
        let input: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, 2, 3, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
                10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            ],
        ));
        let weight: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [2, 1, 2, 2],
            &[
                1.0, 0.0, 0.0, 1.0, //
                0.5, 0.0, 0.0, 0.5,
            ],
        ));

        let conv = Conv2d::new(
            weight,
            None,
            Conv2dConfig {
                groups: 2,
                ..Default::default()
            },
        );
        let output = conv.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2, 2, 2]);
        assert!((result[[0, 0, 0, 0]] - 6.0).abs() < 1e-5);
        assert!((result[[0, 0, 1, 1]] - 14.0).abs() < 1e-5);
        assert!((result[[0, 1, 0, 0]] - 12.0).abs() < 1e-5);
        assert!((result[[0, 1, 1, 1]] - 16.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv2d_asymmetric_kernel_order() {
        let input: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, 1, 3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0,
            ],
        ));
        let weight: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, 1, 2, 2],
            &[
                1.0, 2.0, //
                3.0, 4.0,
            ],
        ));

        let conv = Conv2d::new(weight, None, Conv2dConfig::default());
        let output = conv.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 1, 2, 3]);
        assert!((result[[0, 0, 0, 0]] - 44.0).abs() < 1e-5);
        assert!((result[[0, 0, 0, 1]] - 54.0).abs() < 1e-5);
        assert!((result[[0, 0, 0, 2]] - 64.0).abs() < 1e-5);
        assert!((result[[0, 0, 1, 0]] - 84.0).abs() < 1e-5);
        assert!((result[[0, 0, 1, 1]] - 94.0).abs() < 1e-5);
        assert!((result[[0, 0, 1, 2]] - 104.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv2d_gpu_matches_cpu_stride2_padding1() {
        let gpu = crate::Device::new()
            .await
            .expect("GPU required for this test");

        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0,
        ];
        let weight_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let bias_data = vec![0.25f32];

        let cpu_input: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 4, 4], &input_data));
        let gpu_input: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 1, 4, 4], &input_data);
        let cpu_weight: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2, 2], &weight_data));
        let gpu_weight: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 1, 2, 2], &weight_data);
        let cpu_bias: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1], &bias_data));
        let gpu_bias: Tensor<1, f32> = Tensor::from_slice(&gpu, [1], &bias_data);

        let config = Conv2dConfig {
            padding: [1, 1],
            stride: [2, 2],
            groups: 1,
            dilation: [1, 1],
        };

        let cpu_out = Conv2d::new(cpu_weight, Some(cpu_bias), config)
            .forward(&cpu_input)
            .as_slice()
            .await
            .unwrap();
        let gpu_out = Conv2d::new(gpu_weight, Some(gpu_bias), config)
            .forward(&gpu_input)
            .as_slice()
            .await
            .unwrap();

        for h in 0..cpu_out.shape()[2] {
            for w in 0..cpu_out.shape()[3] {
                let expected = cpu_out[[0, 0, h, w]];
                let actual = gpu_out[[0, 0, h, w]];
                assert!(
                    (expected - actual).abs() < 1e-5,
                    "mismatch at [{h}, {w}]: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_conv2d_gpu_matches_cpu_depthwise_stride2_padding1() {
        let gpu = crate::Device::new()
            .await
            .expect("GPU required for this test");

        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0, 16.0, //
            17.0, 18.0, 19.0, 20.0, //
            21.0, 22.0, 23.0, 24.0, //
            25.0, 26.0, 27.0, 28.0, //
            29.0, 30.0, 31.0, 32.0,
        ];
        let weight_data = vec![
            1.0f32, 0.0, 0.0, 1.0, //
            0.5, 0.0, 0.0, 0.5,
        ];

        let cpu_input: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 4, 4], &input_data));
        let gpu_input: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 2, 4, 4], &input_data);
        let cpu_weight: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 1, 2, 2], &weight_data));
        let gpu_weight: Tensor<4, f32> = Tensor::from_slice(&gpu, [2, 1, 2, 2], &weight_data);

        let config = Conv2dConfig {
            padding: [1, 1],
            stride: [2, 2],
            groups: 2,
            dilation: [1, 1],
        };

        let cpu_out = Conv2d::new(cpu_weight, None, config)
            .forward(&cpu_input)
            .as_slice()
            .await
            .unwrap();
        let gpu_out = Conv2d::new(gpu_weight, None, config)
            .forward(&gpu_input)
            .as_slice()
            .await
            .unwrap();

        for c in 0..cpu_out.shape()[1] {
            for h in 0..cpu_out.shape()[2] {
                for w in 0..cpu_out.shape()[3] {
                    let expected = cpu_out[[0, c, h, w]];
                    let actual = gpu_out[[0, c, h, w]];
                    assert!(
                        (expected - actual).abs() < 1e-5,
                        "mismatch at [{c}, {h}, {w}]: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }
}
