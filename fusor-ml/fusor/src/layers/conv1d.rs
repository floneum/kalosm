//! Conv1d layer implementation.

use crate::{ConcreteTensor, MatmulImpl, SimdElement, Tensor};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::FloatOps;

/// Configuration for Conv1d layer.
#[derive(Debug, Clone, Copy)]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub groups: usize,
    pub dilation: usize,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            groups: 1,
            dilation: 1,
        }
    }
}

/// 1D Convolution layer.
///
/// Applies a 1D convolution over an input signal.
/// Input shape: (batch, in_channels, length)
/// Output shape: (batch, out_channels, out_length)
/// where out_length = (length + 2*padding - kernel_size) / stride + 1
pub struct Conv1d<D: SimdElement> {
    weight: Tensor<3, D, ConcreteTensor<D, 3>>, // (out_channels, in_channels, kernel_size)
    bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>, // (out_channels,)
    config: Conv1dConfig,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl<D> Conv1d<D>
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
    /// Create a new Conv1d layer with given weights and configuration.
    ///
    /// Weight shape: (out_channels, in_channels, kernel_size)
    /// Bias shape: (out_channels,)
    pub fn new(
        weight: Tensor<3, D, ConcreteTensor<D, 3>>,
        bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
        config: Conv1dConfig,
    ) -> Self {
        let shape = weight.shape();
        let out_channels = shape[0];
        let in_channels = shape[1];
        let kernel_size = shape[2];

        // Validate configuration
        assert_eq!(config.groups, 1, "Only groups=1 is currently supported");
        assert_eq!(config.dilation, 1, "Only dilation=1 is currently supported");

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
            kernel_size,
        }
    }

    /// Forward pass.
    ///
    /// Input shape: (batch, in_channels, length)
    /// Output shape: (batch, out_channels, out_length)
    pub fn forward(
        &self,
        input: &Tensor<3, D, ConcreteTensor<D, 3>>,
    ) -> Tensor<3, D, ConcreteTensor<D, 3>>
    where
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
    {
        input.conv(
            &self.weight,
            self.bias.as_ref(),
            [self.config.padding],
            [self.config.stride],
        )
    }

    /// Get the configuration.
    pub fn config(&self) -> &Conv1dConfig {
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
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

#[cfg(test)]
#[allow(clippy::useless_conversion)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conv1d_simple() {
        // Weight: (1, 1, 3) - 1 out channel, 1 in channel, kernel size 3
        let weight_data = [0.2f32, 0.5, 0.3];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &weight_data));

        // Bias: (1,)
        let bias_data = [0.1f32];
        let bias: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1], &bias_data));

        let config = Conv1dConfig::default();
        let conv = Conv1d::new(weight, Some(bias), config);

        // Input: (1, 1, 5) - batch=1, in_channels=1, length=5
        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 5], &input_data));

        let output = conv.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 1, 3]);

        // Manual calculation:
        // output[0] = 1*0.2 + 2*0.5 + 3*0.3 + 0.1 = 2.2
        // output[1] = 2*0.2 + 3*0.5 + 4*0.3 + 0.1 = 3.2
        // output[2] = 3*0.2 + 4*0.5 + 5*0.3 + 0.1 = 4.2
        assert!((result[[0, 0, 0]] - 2.2).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 3.2).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 4.2).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv1d_with_padding() {
        // Weight: (1, 1, 3)
        let weight_data = [1.0f32, 1.0, 1.0];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &weight_data));

        let config = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = Conv1d::new(weight, None, config);

        // Input: (1, 1, 3)
        let input_data = [1.0f32, 2.0, 3.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &input_data));

        let output = conv.forward(&input);
        let result = output.as_slice().await.unwrap();

        // With padding=1, input becomes [0, 1, 2, 3, 0]
        // Output shape: (1, 1, 3)
        assert_eq!(result.shape(), &[1, 1, 3]);

        // output[0] = 0*1 + 1*1 + 2*1 = 3
        // output[1] = 1*1 + 2*1 + 3*1 = 6
        // output[2] = 2*1 + 3*1 + 0*1 = 5
        assert!((result[[0, 0, 0]] - 3.0).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 6.0).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv1d_properties() {
        let weight_data = [0.0f32; 6];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3, 1], &weight_data));

        let config = Conv1dConfig {
            padding: 2,
            stride: 3,
            ..Default::default()
        };
        let conv = Conv1d::new(weight, None, config);

        assert_eq!(conv.in_channels(), 3);
        assert_eq!(conv.out_channels(), 2);
        assert_eq!(conv.kernel_size(), 1);
        assert_eq!(conv.config().padding, 2);
        assert_eq!(conv.config().stride, 3);
    }

    #[tokio::test]
    async fn test_conv1d_cpu_vs_gpu() {
        use crate::Device;

        // Create random-ish weight and input data
        let weight_data: Vec<f32> = (0..384 * 80 * 3)
            .map(|i| (i as f32 * 0.001).sin() * 0.1)
            .collect();
        let bias_data: Vec<f32> = (0..384).map(|i| (i as f32 * 0.0001).cos() * 0.01).collect();
        let input_data: Vec<f32> = (0..80 * 3000).map(|i| (i as f32 * 0.0001).sin()).collect();

        let config = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
        };

        // CPU version
        let weight_cpu: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([384, 80, 3], &weight_data));
        let bias_cpu: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([384], &bias_data));
        let input_cpu: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 80, 3000], &input_data));
        let conv_cpu = Conv1d::new(weight_cpu, Some(bias_cpu), config);
        let output_cpu = conv_cpu.forward(&input_cpu);
        let result_cpu = output_cpu.as_slice().await.unwrap();

        // GPU version
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let weight_gpu: Tensor<3, f32> =
            Tensor::from_slice(&gpu_device, [384, 80, 3], &weight_data);
        let bias_gpu: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [384], &bias_data);
        let input_gpu: Tensor<3, f32> = Tensor::from_slice(&gpu_device, [1, 80, 3000], &input_data);
        let conv_gpu = Conv1d::new(weight_gpu, Some(bias_gpu), config);
        let output_gpu = conv_gpu.forward(&input_gpu);
        let result_gpu = output_gpu.as_slice().await.unwrap();

        // Compare
        assert_eq!(result_cpu.shape(), result_gpu.shape());

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        let mut count = 0;
        for i in 0..result_cpu.shape()[0] {
            for j in 0..result_cpu.shape()[1] {
                for k in 0..result_cpu.shape()[2].min(100) {
                    // Sample first 100 positions
                    let cpu_val: f32 = result_cpu[[i, j, k]].into();
                    let gpu_val: f32 = result_gpu[[i, j, k]].into();
                    let diff = (cpu_val - gpu_val).abs();
                    max_diff = max_diff.max(diff);
                    sum_diff += diff;
                    count += 1;
                }
            }
        }

        eprintln!(
            "Conv1d CPU vs GPU: max_diff={}, mean_diff={}",
            max_diff,
            sum_diff / count as f32
        );
        eprintln!(
            "CPU[0,0,0..5]: {:?}",
            (0..5).map(|i| result_cpu[[0, 0, i]]).collect::<Vec<f32>>()
        );
        eprintln!(
            "GPU[0,0,0..5]: {:?}",
            (0..5).map(|i| result_gpu[[0, 0, i]]).collect::<Vec<f32>>()
        );

        assert!(
            max_diff < 0.01,
            "Conv1d CPU and GPU outputs differ too much: max_diff={}",
            max_diff
        );
    }
}
