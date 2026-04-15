//! RMS normalization implementation.

use crate::{
    CastTensor, CastTo, ConcreteTensor, DataType, Device, SimdElement, Tensor, VarBuilder,
};

/// Root Mean Square Normalization.
///
/// Normalizes the input over the last dimension without centering.
/// Formula: output = input / sqrt(mean(x^2) + eps) * weight
pub struct RmsNorm<const N: usize, T: SimdElement> {
    weight: Tensor<N, T, ConcreteTensor<T, N>>,
    bias: Option<Tensor<N, T, ConcreteTensor<T, N>>>,
    eps: f32,
}

impl<const N: usize, T: DataType + SimdElement + Default> RmsNorm<N, T> {
    /// Create a new RmsNorm layer.
    ///
    /// Weight should have shape matching the normalized dimension.
    pub fn new(weight: Tensor<N, T>, bias: Option<Tensor<N, T>>, eps: f32) -> Self {
        Self {
            weight: weight.to_concrete(),
            bias: bias.map(|b| b.to_concrete()),
            eps,
        }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor<N, T, ConcreteTensor<T, N>> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor<N, T, ConcreteTensor<T, N>>> {
        self.bias.as_ref()
    }

    /// Get the epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Cast the RmsNorm to a different data type
    pub fn cast<U: DataType + SimdElement + Default>(self) -> RmsNorm<N, U>
    where
        T: CastTensor<U> + CastTo<U>,
    {
        RmsNorm {
            weight: self.weight.cast(),
            bias: self.bias.map(|b| b.cast()),
            eps: self.eps,
        }
    }
}

// f32-specific implementations for loading
impl<const R: usize> RmsNorm<R, f32> {
    /// Load RmsNorm from VarBuilder
    pub fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> crate::Result<Self> {
        let weight = vb.get("weight", device)?.dequantize();
        let bias = vb.get("bias", device).ok().map(|b| b.dequantize());
        Ok(Self::new(weight, bias, eps))
    }
}

// Forward implementations for specific ranks (2D, 3D, 4D inputs)
// This avoids the complex trait bounds while still being useful
impl RmsNorm<1, f32> {
    /// Forward pass for 2D input (batch, features).
    pub fn forward_2d<B>(&self, input: &Tensor<2, f32, B>) -> Tensor<2, f32>
    where
        B: fusor_cpu::TensorBacking<2, Elem = f32>,
    {
        input.rms_norm_fused::<1, 1>(&self.weight, self.bias.as_ref(), self.eps)
    }

    /// Forward pass for 3D input (batch, seq_len, features).
    pub fn forward<B>(&self, input: &Tensor<3, f32, B>) -> Tensor<3, f32>
    where
        B: fusor_cpu::TensorBacking<3, Elem = f32>,
    {
        input.rms_norm_fused::<1, 2>(&self.weight, self.bias.as_ref(), self.eps)
    }

    /// Forward pass for 4D input (batch, heads, seq_len, features).
    pub fn forward_4d<B>(&self, input: &Tensor<4, f32, B>) -> Tensor<4, f32>
    where
        B: fusor_cpu::TensorBacking<4, Elem = f32>,
    {
        input.rms_norm_fused::<1, 3>(&self.weight, self.bias.as_ref(), self.eps)
    }
}

// Generic forward implementations for RmsNorm<1, T> where T can be cast to/from f32
// This enables f16 and other types to use RmsNorm by converting to f32 for computation
impl<T: DataType + SimdElement + Default> RmsNorm<1, T>
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
        // Cast input and weights to f32
        let input_f32 = input.cast::<f32>();
        let weight_f32: Tensor<1, f32> = self.weight.cast();
        let bias_f32: Option<Tensor<1, f32>> = self.bias.as_ref().map(|b| b.cast());

        // Compute RMS norm in f32
        let result_f32 = input_f32.rms_norm_fused::<1, 2>(&weight_f32, bias_f32.as_ref(), self.eps);

        // Cast back to T
        result_f32.cast()
    }

    /// Forward pass for 4D input with generic type.
    pub fn forward_generic_4d<B>(&self, input: &Tensor<4, T, B>) -> Tensor<4, T>
    where
        B: fusor_cpu::TensorBacking<4, Elem = T>,
    {
        let input_f32 = input.cast::<f32>();
        let weight_f32: Tensor<1, f32> = self.weight.cast();
        let bias_f32: Option<Tensor<1, f32>> = self.bias.as_ref().map(|b| b.cast());

        let result_f32 = input_f32.rms_norm_fused::<1, 3>(&weight_f32, bias_f32.as_ref(), self.eps);

        result_f32.cast()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rms_norm_2d() {
        // Weight: (3,)
        let weight_data = [1.0f32, 1.0, 1.0];
        let weight: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &weight_data));

        let rms_norm = RmsNorm::new(weight, None, 1e-5);

        // Input: (2, 3)
        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &input_data));

        let output = rms_norm.forward_2d(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[2, 3]);

        // RMS for [1, 2, 3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
        let rms = ((1.0 + 4.0 + 9.0) / 3.0f32).sqrt();
        assert!((result[[0, 0]] - 1.0 / rms).abs() < 1e-4);
        assert!((result[[0, 1]] - 2.0 / rms).abs() < 1e-4);
        assert!((result[[0, 2]] - 3.0 / rms).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_rms_norm_3d() {
        let weight_data = [2.0f32, 2.0];
        let weight: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2], &weight_data));

        let rms_norm = RmsNorm::new(weight, None, 1e-5);

        // Input: (1, 2, 2)
        let input_data = [3.0f32, 4.0, 6.0, 8.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2], &input_data));

        let output = rms_norm.forward(&input);
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2, 2]);

        // RMS for [3, 4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.54
        let rms1 = ((9.0 + 16.0) / 2.0f32).sqrt();
        assert!((result[[0, 0, 0]] - 3.0 / rms1 * 2.0).abs() < 1e-4);
        assert!((result[[0, 0, 1]] - 4.0 / rms1 * 2.0).abs() < 1e-4);
    }
}
