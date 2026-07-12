//! Linear layer implementation.

pub(crate) mod autograd;

use crate::{
    CastTensor, CastTo, DataType, Device, Fusion, GgmlType, QMatrix, SimdElement, Tensor,
    VarBuilder,
};

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

    /// Get the quantized weight matrix.
    pub fn weight(&self) -> &QMatrix {
        &self.weight
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

    /// Applies the linear projection to the last dimension of an input tensor.
    pub fn forward<const R: usize, B>(&self, input: &Tensor<R, f32, B>) -> Tensor<R, f32>
    where
        B: Fusion<R, f32>,
        (crate::gpu::Tensor<R, f32>, crate::gpu::Tensor<1, f32>): crate::gpu::MaxRank<R, f32>,
        (crate::ConcreteTensor<f32, R>, crate::ConcreteTensor<f32, 1>): crate::cpu::MaxRank<R, f32>,
    {
        let output = input.q_mat_mul(&self.weight);
        if let Some(bias) = &self.bias {
            output.add_::<1, R, _>(bias)
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
    /// Applies the linear projection after converting the input to f32.
    pub fn forward_generic<const R: usize, B>(&self, input: &Tensor<R, T, B>) -> Tensor<R, T>
    where
        B: Fusion<R, T>,
        (crate::gpu::Tensor<R, f32>, crate::gpu::Tensor<1, f32>): crate::gpu::MaxRank<R, f32>,
        (crate::ConcreteTensor<f32, R>, crate::ConcreteTensor<f32, 1>): crate::cpu::MaxRank<R, f32>,
    {
        let input_f32 = input.cast::<f32>();
        let output_f32 = input_f32.q_mat_mul(&self.weight);
        let output_f32 = if let Some(bias) = &self.bias {
            let bias_f32: Tensor<1, f32> = bias.cast();
            output_f32.add_::<1, R, _>(&bias_f32)
        } else {
            output_f32
        };
        output_f32.cast()
    }
}
