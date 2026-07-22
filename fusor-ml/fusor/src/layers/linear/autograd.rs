//! Trainable linear layer implementation.

use crate::autograd::{AutogradElement, Graph, Tensor};

/// A trainable linear (fully connected) layer.
///
/// Computes `output = input @ weight.T + bias`.
pub struct Linear<T: AutogradElement = f32> {
    weight: Tensor<2, T>,
    bias: Option<Tensor<1, T>>,
}

impl<T: AutogradElement> Linear<T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
{
    /// Create a new Linear layer with the given weight and optional bias.
    ///
    /// Weight shape: (out_features, in_features)
    /// Bias shape: (out_features,)
    pub fn new(weight: Tensor<2, T>, bias: Option<Tensor<1, T>>) -> Self {
        Self { weight, bias }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor<2, T> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor<1, T>> {
        self.bias.as_ref()
    }

    /// Get the input features size.
    pub fn in_features(&self) -> usize {
        self.weight.shape()[1]
    }

    /// Get the output features size.
    pub fn out_features(&self) -> usize {
        self.weight.shape()[0]
    }
}

impl Linear {
    /// Import an inference layer as trainable graph leaves, dequantizing the weight to f32.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::Linear<f32>) -> Self {
        let weight = graph.leaf(layer.weight().dequantize::<2>().into_concrete());
        let bias = layer.bias().map(|bias| graph.leaf(bias.clone()));
        Self { weight, bias }
    }
}

impl<T: AutogradElement> Linear<T>
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
    /// Applies the linear projection to the last dimension of an input tensor.
    pub fn forward<const R: usize>(&self, input: &Tensor<R, T>) -> Tensor<R, T> {
        assert!(R >= 2, "linear forward requires rank >= 2");

        let input_shape = input.shape();
        let rows = input_shape[..R - 1].iter().product();
        let input_2d = input.reshape([rows, input_shape[R - 1]]);
        let output_2d = input_2d.mat_mul_transposed_rhs(&self.weight);
        let output_2d = if let Some(bias) = &self.bias {
            output_2d.add_(bias)
        } else {
            output_2d
        };
        let output_shape = std::array::from_fn(|axis| {
            if axis == R - 1 {
                self.out_features()
            } else {
                input_shape[axis]
            }
        });
        output_2d.reshape(output_shape)
    }
}
