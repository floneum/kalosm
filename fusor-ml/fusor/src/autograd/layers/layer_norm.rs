//! Trainable layer normalization.

use super::super::{Graph, Tensor};

/// Layer Normalization.
///
/// Normalizes the input over the last dimension.
/// Formula: output = (input - mean) / sqrt(variance + eps) * weight + bias
pub struct LayerNorm<const N: usize> {
    weight: Tensor<N>,
    bias: Option<Tensor<N>>,
    eps: f32,
}

impl<const N: usize> LayerNorm<N> {
    /// Create a new LayerNorm layer.
    ///
    /// Weight and bias should have shape (normalized_dim,).
    pub fn new(weight: Tensor<N>, bias: Option<Tensor<N>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Import an inference [`crate::layers::LayerNorm`]'s weights as trainable leaves.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::LayerNorm<N, f32>) -> Self {
        Self::new(
            graph.leaf(layer.weight().clone()),
            layer.bias().map(|bias| graph.leaf(bias.clone())),
            layer.eps(),
        )
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

impl LayerNorm<1> {
    /// Forward pass for 2D input (batch, features).
    ///
    /// Normalizes over the last dimension (features).
    pub fn forward_2d(&self, input: &Tensor<2>) -> Tensor<2> {
        let weight_broadcast = self.weight.broadcast_as(input.shape());
        let bias_broadcast = self
            .bias
            .as_ref()
            .map(|bias| bias.broadcast_as(input.shape()));
        input.layer_norm::<1>(&weight_broadcast, bias_broadcast.as_ref(), self.eps, true)
    }

    /// Forward pass for 3D input (batch, seq_len, features).
    ///
    /// Normalizes over the last dimension (features).
    pub fn forward(&self, input: &Tensor<3>) -> Tensor<3> {
        let weight_broadcast = self.weight.broadcast_as(input.shape());
        let bias_broadcast = self
            .bias
            .as_ref()
            .map(|bias| bias.broadcast_as(input.shape()));
        input.layer_norm::<2>(&weight_broadcast, bias_broadcast.as_ref(), self.eps, true)
    }

    /// Forward pass through the fused last-dim kernel (3D input).
    pub fn forward_fused(&self, input: &Tensor<3>) -> Tensor<3> {
        input.layer_norm_last_dim_fused::<2, 1>(&self.weight, self.bias.as_ref(), self.eps)
    }
}

/// Layer normalization with a selectable reduction axis.
///
/// `axis == None` normalizes the last dimension. `axis == Some(a)` normalizes
/// dimension `a` by transposing that axis to the end, applying the common
/// last-dimension path, then transposing back.
pub struct LayerNormNd {
    weight: Tensor<1>,
    bias: Option<Tensor<1>>,
    axis: Option<usize>,
    eps: f32,
}

impl LayerNormNd {
    /// Create a LayerNorm that normalizes the last dimension.
    pub fn new(weight: Tensor<1>, bias: Option<Tensor<1>>, eps: f32) -> Self {
        Self {
            weight,
            bias,
            axis: None,
            eps,
        }
    }

    /// Create a LayerNorm that normalizes the given axis.
    pub fn new_over_axis(
        weight: Tensor<1>,
        bias: Option<Tensor<1>>,
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

    /// Import an inference [`crate::layers::LayerNormNd`]'s weights as trainable
    /// leaves, normalizing the last dimension.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::LayerNormNd<f32>) -> Self {
        Self::new(
            graph.leaf(layer.weight().clone()),
            layer.bias().map(|bias| graph.leaf(bias.clone())),
            layer.eps(),
        )
    }

    /// Import an inference [`crate::layers::LayerNormNd`]'s weights as trainable
    /// leaves, normalizing `axis`.
    pub fn from_inference_over_axis(
        graph: &Graph,
        layer: &crate::layers::LayerNormNd<f32>,
        axis: usize,
    ) -> Self {
        Self::new_over_axis(
            graph.leaf(layer.weight().clone()),
            layer.bias().map(|bias| graph.leaf(bias.clone())),
            axis,
            layer.eps(),
        )
    }

    pub fn weight(&self) -> &Tensor<1> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<1>> {
        self.bias.as_ref()
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Forward pass for 2D input.
    pub fn forward_2d(&self, input: &Tensor<2>) -> Tensor<2> {
        self.forward::<2, 1>(input)
    }

    /// Forward pass for any input rank. `OUT_RANK` equals `N - 1`.
    pub fn forward<const N: usize, const OUT_RANK: usize>(&self, input: &Tensor<N>) -> Tensor<N>
    where
        crate::ConcreteTensor<f32, N>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<N, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        let shape = input.shape();
        let axis = self.axis.unwrap_or(N - 1);

        if axis == N - 1 {
            let weight_b = self.weight.broadcast_as(shape);
            let bias_b = self.bias.as_ref().map(|bias| bias.broadcast_as(shape));
            return input.layer_norm::<OUT_RANK>(&weight_b, bias_b.as_ref(), self.eps, true);
        }

        let mut permuted_shape = shape;
        permuted_shape.swap(axis, N - 1);
        let permuted = input.transpose(axis, N - 1);
        let weight_b = self.weight.broadcast_as(permuted_shape);
        let bias_b = self
            .bias
            .as_ref()
            .map(|bias| bias.broadcast_as(permuted_shape));
        let normed = permuted.layer_norm::<OUT_RANK>(&weight_b, bias_b.as_ref(), self.eps, true);
        normed.transpose(axis, N - 1)
    }

    /// Fused fast path for normalizing the last dim of a rank-3 tensor.
    pub fn forward_fused(&self, input: &Tensor<3>) -> Tensor<3> {
        if matches!(self.axis, Some(axis) if axis != 2) {
            return self.forward::<3, 2>(input);
        }

        input.layer_norm_last_dim_fused::<2, 1>(&self.weight, self.bias.as_ref(), self.eps)
    }
}
