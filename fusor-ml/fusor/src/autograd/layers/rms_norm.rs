//! Trainable RMS normalization implementation.

use super::super::{Graph, Tensor};

/// Root Mean Square Normalization.
///
/// Normalizes the input over the last dimension without centering.
/// Formula: output = input / sqrt(mean(x^2) + eps) * weight
pub struct RmsNorm<const N: usize> {
    weight: Tensor<N>,
    bias: Option<Tensor<N>>,
    eps: f32,
}

impl<const N: usize> RmsNorm<N> {
    /// Create a new RmsNorm layer.
    ///
    /// Weight should have shape matching the normalized dimension.
    pub fn new(weight: Tensor<N>, bias: Option<Tensor<N>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Import a raw inference layer's weights as trainable graph leaves.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::RmsNorm<N, f32>) -> Self {
        Self {
            weight: graph.leaf(layer.weight().clone()),
            bias: layer.bias().map(|bias| graph.leaf(bias.clone())),
            eps: layer.eps(),
        }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor<N> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor<N>> {
        self.bias.as_ref()
    }

    /// Get the epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl RmsNorm<1> {
    /// Forward pass for 2D input (batch, features).
    pub fn forward_2d(&self, input: &Tensor<2>) -> Tensor<2> {
        input.rms_norm_fused::<1, 1>(&self.weight, self.bias.as_ref(), self.eps)
    }

    /// Forward pass for 3D input (batch, seq_len, features).
    pub fn forward(&self, input: &Tensor<3>) -> Tensor<3> {
        input.rms_norm_fused::<1, 2>(&self.weight, self.bias.as_ref(), self.eps)
    }

    /// Forward pass for 4D input (batch, heads, seq_len, features).
    pub fn forward_4d(&self, input: &Tensor<4>) -> Tensor<4> {
        input.rms_norm_fused::<1, 3>(&self.weight, self.bias.as_ref(), self.eps)
    }

    /// Forward pass for `input + residual` followed by RMSNorm.
    pub fn forward_residual(&self, input: &Tensor<3>, residual: &Tensor<3>) -> Tensor<3> {
        input.rms_norm_residual_fused::<1, 2>(residual, &self.weight, self.bias.as_ref(), self.eps)
    }
}

