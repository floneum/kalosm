//! Trainable layer normalization.

use crate::autograd::{AutogradElement, Graph, Tensor};

/// Layer Normalization.
///
/// Normalizes the input over the last dimension.
/// Formula: output = (input - mean) / sqrt(variance + eps) * weight + bias
pub struct LayerNorm<const N: usize, T: AutogradElement = f32> {
    weight: Tensor<N, T>,
    bias: Option<Tensor<N, T>>,
    eps: f32,
}

impl<const N: usize, T: AutogradElement> LayerNorm<N, T> {
    /// Create a new LayerNorm layer.
    ///
    /// Weight and bias should have shape (normalized_dim,).
    pub fn new(weight: Tensor<N, T>, bias: Option<Tensor<N, T>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor<N, T> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor<N, T>> {
        self.bias.as_ref()
    }

    /// Get the epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl<const N: usize> LayerNorm<N> {
    /// Import an inference [`crate::layers::LayerNorm`]'s weights as trainable leaves.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::LayerNorm<N, f32>) -> Self {
        Self::new(
            graph.leaf(layer.weight().clone()),
            layer.bias().map(|bias| graph.leaf(bias.clone())),
            layer.eps(),
        )
    }
}

impl<T: AutogradElement> LayerNorm<1, T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
{
    /// Normalizes the last dimension of an input tensor.
    pub fn forward<const R: usize, const OUT_RANK: usize>(
        &self,
        input: &Tensor<R, T>,
    ) -> Tensor<R, T>
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
    {
        input.layer_norm_last_dim_fused::<OUT_RANK, 1>(&self.weight, self.bias.as_ref(), self.eps)
    }
}

/// Layer normalization with a selectable reduction axis.
///
/// `axis == None` normalizes the last dimension. `axis == Some(a)` normalizes
/// dimension `a` by transposing that axis to the end, applying the common
/// last-dimension path, then transposing back.
pub struct LayerNormNd<T: AutogradElement = f32> {
    weight: Tensor<1, T>,
    bias: Option<Tensor<1, T>>,
    axis: Option<usize>,
    eps: f32,
}

impl<T: AutogradElement> LayerNormNd<T> {
    /// Create a LayerNorm that normalizes the last dimension.
    pub fn new(weight: Tensor<1, T>, bias: Option<Tensor<1, T>>, eps: f32) -> Self {
        Self {
            weight,
            bias,
            axis: None,
            eps,
        }
    }

    /// Create a LayerNorm that normalizes the given axis.
    pub fn new_over_axis(
        weight: Tensor<1, T>,
        bias: Option<Tensor<1, T>>,
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

    pub fn weight(&self) -> &Tensor<1, T> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<1, T>> {
        self.bias.as_ref()
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl LayerNormNd {
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
}

impl<T: AutogradElement> LayerNormNd<T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
{
    /// Forward pass for any input rank. `OUT_RANK` equals `N - 1`.
    pub fn forward<const N: usize, const OUT_RANK: usize>(
        &self,
        input: &Tensor<N, T>,
    ) -> Tensor<N, T>
    where
        crate::ConcreteTensor<T, N>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<N, T>: crate::gpu::LastRank<OUT_RANK, T>,
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
    pub fn forward_fused(&self, input: &Tensor<3, T>) -> Tensor<3, T> {
        if matches!(self.axis, Some(axis) if axis != 2) {
            return self.forward::<3, 2>(input);
        }

        input.layer_norm_last_dim_fused::<2, 1>(&self.weight, self.bias.as_ref(), self.eps)
    }
}
