//! Trainable linear layer implementation.

use super::super::{Graph, Tensor};

/// A trainable linear (fully connected) layer.
///
/// Computes `output = input @ weight.T + bias`.
pub struct Linear {
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
}

impl Linear {
    /// Create a new Linear layer with the given weight and optional bias.
    ///
    /// Weight shape: (out_features, in_features)
    /// Bias shape: (out_features,)
    pub fn new(weight: Tensor<2>, bias: Option<Tensor<1>>) -> Self {
        Self { weight, bias }
    }

    /// Import an inference layer as trainable graph leaves, dequantizing the weight to f32.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::Linear<f32>) -> Self {
        let weight = graph.leaf(layer.weight().dequantize::<2>().to_concrete());
        let bias = layer.bias().map(|bias| graph.leaf(bias.clone()));
        Self { weight, bias }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor<2> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor<1>> {
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

    /// Forward pass for 3D input (batch, seq_len, in_features)
    ///
    /// Input shape: (batch, seq_len, in_features)
    /// Output shape: (batch, seq_len, out_features)
    pub fn forward(&self, input: &Tensor<3>) -> Tensor<3> {
        let [batch, seq_len, in_features] = input.shape();
        let input_2d = input.reshape([batch * seq_len, in_features]);
        self.forward_2d(&input_2d)
            .reshape([batch, seq_len, self.out_features()])
    }

    /// Forward pass for 2D input (batch, in_features)
    ///
    /// Input shape: (batch, in_features)
    /// Output shape: (batch, out_features)
    pub fn forward_2d(&self, input: &Tensor<2>) -> Tensor<2> {
        let output = input.mat_mul_transposed_rhs(&self.weight);
        if let Some(bias) = &self.bias {
            output.add_(bias)
        } else {
            output
        }
    }
}
