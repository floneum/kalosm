//! Trainable RMS normalization implementation.

use crate::autograd::{Graph, Tensor};

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
    /// Normalizes the last dimension of an input tensor.
    pub fn forward<const R: usize, const OUT_RANK: usize>(&self, input: &Tensor<R>) -> Tensor<R>
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, f32> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, f32>>,
        (crate::gpu::Tensor<R, f32>, crate::gpu::Tensor<1, f32>): crate::gpu::MaxRank<R, f32>,
    {
        input.rms_norm_fused::<1, OUT_RANK>(&self.weight, self.bias.as_ref(), self.eps)
    }

    /// Forward pass for `input + residual` followed by RMSNorm.
    pub fn forward_residual(&self, input: &Tensor<3>, residual: &Tensor<3>) -> Tensor<3> {
        input.rms_norm_residual_fused::<1, 2>(residual, &self.weight, self.bias.as_ref(), self.eps)
    }
}
