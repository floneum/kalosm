//! Trainable Conv1d layer implementation.

use super::super::{Graph, Tensor};

pub use crate::layers::Conv1dConfig;

/// 1D Convolution layer with trainable parameters.
///
/// Applies a 1D convolution over an input signal.
/// Input shape: (batch, in_channels, length)
/// Output shape: (batch, out_channels, out_length)
/// where out_length = (length + 2*padding - kernel_size) / stride + 1
pub struct Conv1d {
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    config: Conv1dConfig,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
}

impl Conv1d {
    /// Create a new Conv1d layer with given weights and configuration.
    ///
    /// Weight shape: (out_channels, in_channels, kernel_size)
    /// Bias shape: (out_channels,)
    pub fn new(weight: Tensor<3>, bias: Option<Tensor<1>>, config: Conv1dConfig) -> Self {
        let shape = weight.shape();
        let out_channels = shape[0];
        let in_channels = shape[1];
        let kernel_size = shape[2];

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

    /// Import an inference layer's parameters as trainable graph leaves.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::Conv1d<f32>) -> Self {
        Self::new(
            graph.leaf(layer.weight().clone()),
            layer.bias().map(|bias| graph.leaf(bias.clone())),
            *layer.config(),
        )
    }

    /// Forward pass.
    ///
    /// Input shape: (batch, in_channels, length)
    /// Output shape: (batch, out_channels, out_length)
    pub fn forward(&self, input: &Tensor<3>) -> Tensor<3> {
        input.conv(
            &self.weight,
            self.bias.as_ref(),
            [self.config.padding],
            [self.config.stride],
        )
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor<3> {
        &self.weight
    }

    /// Get the bias tensor.
    pub fn bias(&self) -> Option<&Tensor<1>> {
        self.bias.as_ref()
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

