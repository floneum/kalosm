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

