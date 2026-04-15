//! Linear layer implementation.

use crate::{CastTensor, CastTo, DataType, Device, QMatrix, SimdElement, Tensor, VarBuilder};
use fusor_cpu::GgmlType;

/// A linear (fully connected) layer with quantized weights.
///
/// Computes `output = input @ weight.T + bias` using quantized matrix multiplication.
pub struct Linear<T: SimdElement> {
    weight: QMatrix,
    bias: Option<Tensor<1, T>>,
}

impl<T: DataType + SimdElement + Default> Linear<T> {
    /// Create a new Linear layer with the given quantized weight and optional bias.
    ///
    /// Weight shape: (out_features, in_features)
    /// Bias shape: (out_features,)
    pub fn new(weight: QMatrix, bias: Option<Tensor<1, T>>) -> Self {
        Self { weight, bias }
    }

    /// Get the quantization type of the weights.
    pub fn quantization(&self) -> GgmlType {
        self.weight.ggml_type()
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

    /// Cast the Linear layer to a different data type
    pub fn cast<U: DataType + SimdElement + Default>(self) -> Linear<U>
    where
        T: CastTensor<U> + CastTo<U>,
    {
        Linear {
            weight: self.weight,
            bias: self.bias.map(|b| b.cast()),
        }
    }
}

// f32-specific implementations for loading and forward
impl Linear<f32> {
    /// Load a Linear layer from a VarBuilder.
    ///
    /// Expects:
    /// - weight: Quantized tensor with shape (out_features, in_features)
    /// - bias (optional): Tensor with shape (out_features,)
    pub fn load(device: &Device, vb: &mut VarBuilder) -> crate::Result<Self> {
        let weight = vb.get("weight", device)?;
        let bias: Option<Tensor<1, f32>> = vb.get("bias", device).ok().map(|b| b.dequantize());
        Ok(Self { weight, bias })
    }

    /// Forward pass for 3D input (batch, seq_len, in_features)
    ///
    /// Input shape: (batch, seq_len, in_features)
    /// Output shape: (batch, seq_len, out_features)
    pub fn forward<B>(&self, input: &Tensor<3, f32, B>) -> Tensor<3, f32>
    where
        B: fusor_cpu::TensorBacking<3, Elem = f32>,
    {
        let output = input.q_mat_mul(&self.weight);

        if let Some(bias) = &self.bias {
            output.add_(bias)
        } else {
            output
        }
    }
}

// Generic forward implementations for Linear<T> where T can be cast to/from f32
// This enables f16 and other types to use Linear by converting to f32 for computation
impl<T: DataType + SimdElement + Default> Linear<T>
where
    T: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<T> + CastTensor<T>,
{
    /// Forward pass for 3D input with generic type.
    /// Converts input to f32 for computation, then converts back.
    pub fn forward_generic<B>(&self, input: &Tensor<3, T, B>) -> Tensor<3, T>
    where
        B: fusor_cpu::TensorBacking<3, Elem = T>,
    {
        // Cast input to f32
        let input_f32 = input.cast::<f32>();

        // Do quantized matmul in f32
        let output_f32 = input_f32.q_mat_mul(&self.weight);

        // Add bias if present (in f32)
        let output_f32 = if let Some(bias) = &self.bias {
            let bias_f32: Tensor<1, f32> = bias.cast();
            output_f32.add_(&bias_f32)
        } else {
            output_f32
        };

        // Cast back to T
        output_f32.cast()
    }
}
