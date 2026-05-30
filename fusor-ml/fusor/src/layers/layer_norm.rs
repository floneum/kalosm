//! Layer normalization implementation.

use crate::fusion::Concrete;
use crate::{
    DataType, Device, DivOp, FloatDataType, FloatOps, Fusion, MulOp, SimdBinaryOp, SimdElement,
    SimdReduceOp, SimdUnaryOp, SqrtOp, SubOp, SumOp, Tensor, VarBuilder,
};

fn dequantize_vector_f32(
    tensor: crate::QMatrix,
    name: &str,
) -> crate::Result<Tensor<1, f32, Concrete<f32, 1>>> {
    match tensor.shape().len() {
        1 => {
            let tensor: Tensor<1, f32> = tensor.dequantize();
            Ok(tensor.to_concrete())
        }
        2 => {
            let tensor: Tensor<2, f32> = tensor.dequantize();
            let shape = tensor.shape();
            if shape[0] == 1 {
                Ok(tensor.squeeze(0).to_concrete())
            } else if shape[1] == 1 {
                Ok(tensor.squeeze(1).to_concrete())
            } else {
                Err(crate::Error::VarBuilder(format!(
                    "{name} must be a vector or squeezed vector, got shape {shape:?}",
                )))
            }
        }
        rank => Err(crate::Error::VarBuilder(format!(
            "{name} must be rank 1 or 2, got rank {rank}",
        ))),
    }
}

fn load_vector_f32(
    device: &Device,
    vb: &mut VarBuilder,
    name: &str,
) -> crate::Result<Tensor<1, f32, Concrete<f32, 1>>> {
    let tensor = vb.get(name, device)?;
    dequantize_vector_f32(tensor, name)
}

fn load_optional_vector_f32(
    device: &Device,
    vb: &mut VarBuilder,
    name: &str,
) -> crate::Result<Option<Tensor<1, f32, Concrete<f32, 1>>>> {
    let Ok(tensor) = vb.get(name, device) else {
        return Ok(None);
    };
    dequantize_vector_f32(tensor, name).map(Some)
}

/// Layer Normalization.
///
/// Normalizes the input over the last dimension.
/// Formula: output = (input - mean) / sqrt(variance + eps) * weight + bias
pub struct LayerNorm<const N: usize, D: SimdElement> {
    weight: Tensor<N, D, Concrete<D, N>>,
    bias: Option<Tensor<N, D, Concrete<D, N>>>,
    eps: f32,
}

impl<const N: usize, D> LayerNorm<N, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Create a new LayerNorm layer.
    ///
    /// Weight and bias should have shape (normalized_dim,).
    pub fn new(
        weight: Tensor<N, D, Concrete<D, N>>,
        bias: Option<Tensor<N, D, Concrete<D, N>>>,
        eps: f32,
    ) -> Self {
        Self { weight, bias, eps }
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor<N, D, Concrete<D, N>> {
        &self.weight
    }

    /// Get the bias tensor if present.
    pub fn bias(&self) -> Option<&Tensor<N, D, Concrete<D, N>>> {
        self.bias.as_ref()
    }

    /// Get the epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl<D> LayerNorm<1, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Forward pass for 2D input (batch, features).
    ///
    /// Normalizes over the last dimension (features).
    pub fn forward_2d<B>(&self, input: &Tensor<2, D, B>) -> Tensor<2, D, Concrete<D, 2>>
    where
        D: std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>,
        crate::AddOp: SimdBinaryOp<D>,
        SubOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        SumOp: SimdReduceOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        B: Fusion<2, D>,
    {
        // Broadcast weight to input shape
        let weight_broadcast: Tensor<2, D, _> = self.weight.broadcast_as(input.shape());
        let bias_broadcast: Option<Tensor<2, D, _>> =
            self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));
        input.layer_norm(
            &weight_broadcast,
            bias_broadcast.as_ref(),
            D::from_f32(self.eps),
            true,
        )
    }

    /// Forward pass for 3D input (batch, seq_len, features).
    ///
    /// Normalizes over the last dimension (features).
    pub fn forward<B>(&self, input: &Tensor<3, D, B>) -> Tensor<3, D, Concrete<D, 3>>
    where
        D: std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>,
        crate::AddOp: SimdBinaryOp<D>,
        SubOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        SumOp: SimdReduceOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        B: Fusion<3, D>,
    {
        // Broadcast weight to input shape
        let weight_broadcast: Tensor<3, D, _> = self.weight.broadcast_as(input.shape());
        let bias_broadcast: Option<Tensor<3, D, _>> =
            self.bias.as_ref().map(|b| b.broadcast_as(input.shape()));
        input.layer_norm(
            &weight_broadcast,
            bias_broadcast.as_ref(),
            D::from_f32(self.eps),
            true,
        )
    }
}

impl LayerNorm<1, f32> {
    /// Forward pass with fused CPU kernel (3D input).
    ///
    /// This is significantly faster than the standard forward pass on CPU
    /// as it computes mean, variance, and normalization in fewer passes.
    pub fn forward_fused<B>(&self, input: &Tensor<3, f32, B>) -> Tensor<3, f32>
    where
        B: Fusion<3, f32>,
    {
        input.layer_norm_last_dim_fused::<2, 1, _, _>(&self.weight, self.bias.as_ref(), self.eps)
    }

    /// Load LayerNorm from VarBuilder.
    ///
    /// Expects:
    /// - weight: Tensor with shape matching the normalized dimension
    /// - bias (optional): Tensor with same shape as weight
    pub fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> crate::Result<Self> {
        let weight = load_vector_f32(device, vb, "weight")?;
        let bias = load_optional_vector_f32(device, vb, "bias")?;
        Ok(Self::new(weight, bias, eps))
    }
}

/// Layer normalization with a selectable reduction axis.
///
/// `axis == None` normalizes the last dimension. `axis == Some(a)` normalizes
/// dimension `a` by transposing that axis to the end, applying the common
/// last-dimension path, then transposing back.
pub struct LayerNormNd<D: SimdElement = f32> {
    weight: Tensor<1, D, Concrete<D, 1>>,
    bias: Option<Tensor<1, D, Concrete<D, 1>>>,
    axis: Option<usize>,
    eps: f32,
}

impl<D> LayerNormNd<D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Create a LayerNorm that normalizes the last dimension.
    pub fn new(
        weight: Tensor<1, D, Concrete<D, 1>>,
        bias: Option<Tensor<1, D, Concrete<D, 1>>>,
        eps: f32,
    ) -> Self {
        Self {
            weight,
            bias,
            axis: None,
            eps,
        }
    }

    /// Create a LayerNorm that normalizes the given axis.
    pub fn new_over_axis(
        weight: Tensor<1, D, Concrete<D, 1>>,
        bias: Option<Tensor<1, D, Concrete<D, 1>>>,
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

    pub fn weight(&self) -> &Tensor<1, D, Concrete<D, 1>> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<1, D, Concrete<D, 1>>> {
        self.bias.as_ref()
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Forward pass for 2D input.
    pub fn forward_2d<B>(&self, input: &Tensor<2, D, B>) -> Tensor<2, D, Concrete<D, 2>>
    where
        D: std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>,
        crate::AddOp: SimdBinaryOp<D>,
        SubOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        SumOp: SimdReduceOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        B: Fusion<2, D>,
    {
        self.forward(input)
    }

    /// Forward pass for any input rank. `OUT_RANK` equals `N - 1`.
    pub fn forward<const N: usize, const OUT_RANK: usize, B>(
        &self,
        input: &Tensor<N, D, B>,
    ) -> Tensor<N, D, Concrete<D, N>>
    where
        B: Fusion<N, D>,
        D: std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>
            + std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>,
        crate::AddOp: SimdBinaryOp<D>,
        SubOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        SumOp: SimdReduceOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        Concrete<D, N>: crate::cpu::LastRank<OUT_RANK, D>,
        crate::gpu::Tensor<N, D>: crate::gpu::LastRank<OUT_RANK, D>,
        <crate::gpu::Tensor<N, D> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<N, D>>,
    {
        let shape = input.shape();
        let axis = self.axis.unwrap_or(N - 1);

        if axis == N - 1 {
            let weight_b: Tensor<N, D, _> = self.weight.broadcast_as(shape);
            let bias_b: Option<Tensor<N, D, _>> = self.bias.as_ref().map(|b| b.broadcast_as(shape));
            return input.layer_norm(&weight_b, bias_b.as_ref(), D::from_f32(self.eps), true);
        }

        let mut permuted_shape = shape;
        permuted_shape.swap(axis, N - 1);
        let permuted = input.transpose(axis, N - 1).to_concrete();
        let weight_b: Tensor<N, D, _> = self.weight.broadcast_as(permuted_shape);
        let bias_b: Option<Tensor<N, D, _>> =
            self.bias.as_ref().map(|b| b.broadcast_as(permuted_shape));
        let normed: Tensor<N, D> =
            permuted.layer_norm(&weight_b, bias_b.as_ref(), D::from_f32(self.eps), true);
        normed.transpose(axis, N - 1).to_concrete()
    }
}

impl LayerNormNd<f32> {
    /// Load a last-dim LayerNorm from a `VarBuilder`. Bias is optional.
    pub fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> crate::Result<Self> {
        let weight = load_vector_f32(device, vb, "weight")?;
        let bias = load_optional_vector_f32(device, vb, "bias")?;
        Ok(Self::new(weight, bias, eps))
    }

    /// Load a LayerNorm that normalizes `axis`. Bias is optional.
    pub fn load_over_axis(
        device: &Device,
        vb: &mut VarBuilder,
        axis: usize,
        eps: f32,
    ) -> crate::Result<Self> {
        let weight = load_vector_f32(device, vb, "weight")?;
        let bias = load_optional_vector_f32(device, vb, "bias")?;
        Ok(Self::new_over_axis(weight, bias, axis, eps))
    }

    /// Fused CPU fast path for normalizing the last dim of a rank-3 tensor.
    pub fn forward_fused<B>(&self, input: &Tensor<3, f32, B>) -> Tensor<3, f32>
    where
        B: Fusion<3, f32>,
    {
        if matches!(self.axis, Some(axis) if axis != 2) {
            return self.forward(input);
        }

        input.layer_norm_last_dim_fused::<2, 1, _, _>(&self.weight, self.bias.as_ref(), self.eps)
    }
}
