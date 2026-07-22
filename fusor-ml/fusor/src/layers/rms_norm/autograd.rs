//! Trainable RMS normalization implementation.

use crate::autograd::{AutogradElement, Graph, Tensor};

/// Root Mean Square Normalization.
///
/// Normalizes the input over the last dimension without centering.
/// Formula: output = input / sqrt(mean(x^2) + eps) * weight
pub struct RmsNorm<const N: usize, T: AutogradElement = f32> {
    weight: Tensor<N, T>,
    bias: Option<Tensor<N, T>>,
    eps: f32,
}

impl<const N: usize, T: AutogradElement> RmsNorm<N, T> {
    /// Create a new RmsNorm layer.
    ///
    /// Weight should have shape matching the normalized dimension.
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

impl<const N: usize> RmsNorm<N> {
    /// Import a raw inference layer's weights as trainable graph leaves.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::RmsNorm<N, f32>) -> Self {
        Self {
            weight: graph.leaf(layer.weight().clone()),
            bias: layer.bias().map(|bias| graph.leaf(bias.clone())),
            eps: layer.eps(),
        }
    }
}

impl<T: AutogradElement> RmsNorm<1, T>
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
    T: crate::CastTensor<f32>,
    f32: crate::CastTensor<T>,
{
    /// Normalizes the last dimension of an input tensor.
    pub fn forward<const R: usize, const OUT_RANK: usize>(&self, input: &Tensor<R, T>) -> Tensor<R, T>
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        (crate::gpu::Tensor<R, T>, crate::gpu::Tensor<1, T>): crate::gpu::MaxRank<R, T>,
    {
        input.rms_norm_fused::<1, OUT_RANK>(&self.weight, self.bias.as_ref(), self.eps)
    }

    /// Forward pass for `input + residual` followed by RMSNorm.
    pub fn forward_residual(&self, input: &Tensor<3, T>, residual: &Tensor<3, T>) -> Tensor<3, T> {
        input.rms_norm_residual_fused::<1, 2>(residual, &self.weight, self.bias.as_ref(), self.eps)
    }
}
