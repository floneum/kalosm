//! RMS normalization implementation.

pub(crate) mod autograd;

use crate::fusion::Concrete;
use crate::{CastTensor, CastTo, DataType, Device, Fusion, SimdElement, Tensor, VarBuilder};

/// Root Mean Square Normalization.
///
/// Normalizes the input over the last dimension without centering.
/// Formula: output = input / sqrt(mean(x^2) + eps) * weight
pub struct RmsNorm<const N: usize, T: SimdElement> {
    weight: Tensor<N, T, Concrete<T, N>>,
    bias: Option<Tensor<N, T, Concrete<T, N>>>,
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
    pub fn weight(&self) -> &Tensor<N, T, Concrete<T, N>> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor<N, T, Concrete<T, N>>> {
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

impl RmsNorm<1, f32> {
    /// Normalizes the last dimension of an input tensor.
    pub fn forward<const R: usize, const OUT_RANK: usize, B>(
        &self,
        input: &Tensor<R, f32, B>,
    ) -> Tensor<R, f32>
    where
        B: Fusion<R, f32>,
        Concrete<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, f32> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, f32>>,
        (crate::gpu::Tensor<R, f32>, crate::gpu::Tensor<1, f32>): crate::gpu::MaxRank<R, f32>,
    {
        input.rms_norm_fused::<1, OUT_RANK>(&self.weight, self.bias.as_ref(), self.eps)
    }
}

// Generic forward implementations for RmsNorm<1, T> where T can be cast to/from f32
// This enables f16 and other types to use RmsNorm by converting to f32 for computation
impl<T: DataType + SimdElement + Default> RmsNorm<1, T>
where
    T: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<T> + CastTensor<T>,
{
    /// Normalizes after converting the input to f32, then converts it back.
    pub fn forward_generic<const R: usize, const OUT_RANK: usize, B>(
        &self,
        input: &Tensor<R, T, B>,
    ) -> Tensor<R, T>
    where
        B: Fusion<R, T>,
        Concrete<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, f32> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, f32>>,
        (crate::gpu::Tensor<R, f32>, crate::gpu::Tensor<1, f32>): crate::gpu::MaxRank<R, f32>,
    {
        // Cast input and weights to f32
        let input_f32 = input.cast::<f32>();
        let weight_f32: Tensor<1, f32> = self.weight.cast();
        let bias_f32: Option<Tensor<1, f32>> = self.bias.as_ref().map(|b| b.cast());

        // Compute RMS norm in f32
        let result_f32 =
            input_f32.rms_norm_fused::<1, OUT_RANK>(&weight_f32, bias_f32.as_ref(), self.eps);

        // Cast back to T
        result_f32.cast()
    }

    /// Forward pass for `input + residual` followed by RMSNorm.
    pub fn forward_residual_generic<B1, B2>(
        &self,
        input: &Tensor<3, T, B1>,
        residual: &Tensor<3, T, B2>,
    ) -> Tensor<3, T>
    where
        B1: Fusion<3, T>,
        B2: Fusion<3, T>,
    {
        let input_f32 = input.cast::<f32>();
        let residual_f32 = residual.cast::<f32>();
        let weight_f32: Tensor<1, f32> = self.weight.cast();
        let bias_f32: Option<Tensor<1, f32>> = self.bias.as_ref().map(|b| b.cast());

        let result_f32 = input_f32.rms_norm_residual_fused::<1, 2, _>(
            &residual_f32,
            &weight_f32,
            bias_f32.as_ref(),
            self.eps,
        );

        result_f32.cast()
    }

    /// Forward pass for f32 `input + residual` followed by RMSNorm, returning this layer's type.
    pub fn forward_residual_f32<B1, B2>(
        &self,
        input: &Tensor<3, f32, B1>,
        residual: &Tensor<3, f32, B2>,
    ) -> Tensor<3, T>
    where
        B1: Fusion<3, f32>,
        B2: Fusion<3, f32>,
    {
        let weight_f32: Tensor<1, f32> = self.weight.cast();
        let bias_f32: Option<Tensor<1, f32>> = self.bias.as_ref().map(|b| b.cast());

        input
            .rms_norm_residual_fused::<1, 2, _>(residual, &weight_f32, bias_f32.as_ref(), self.eps)
            .cast()
    }
}
