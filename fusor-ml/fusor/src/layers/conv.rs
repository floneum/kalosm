//! N-dimensional convolution layer.

use crate::fusion::Concrete;
use crate::{
    DataType, Device, FloatDataType, FloatOps, Fusion, MatmulImpl, SimdElement, Tensor, VarBuilder,
};

/// Configuration for an N-D convolution.
#[derive(Debug, Clone, Copy)]
pub struct ConvNdConfig<const SPATIAL: usize> {
    pub padding: [usize; SPATIAL],
    pub stride: [usize; SPATIAL],
    pub groups: usize,
}

impl<const SPATIAL: usize> Default for ConvNdConfig<SPATIAL> {
    fn default() -> Self {
        Self {
            padding: [0; SPATIAL],
            stride: [1; SPATIAL],
            groups: 1,
        }
    }
}

/// N-dimensional convolution layer.
///
/// Input / output tensors have rank `RANK = SPATIAL + 2`:
/// `(batch, channels, ...spatial)`.
/// Weight has shape `(out_channels, in_channels / groups, ...kernel)`.
pub struct ConvNd<const SPATIAL: usize, const RANK: usize, D: SimdElement> {
    weight: Tensor<RANK, D, Concrete<D, RANK>>,
    bias: Option<Tensor<1, D, Concrete<D, 1>>>,
    config: ConvNdConfig<SPATIAL>,
    in_channels: usize,
    out_channels: usize,
}

impl<const SPATIAL: usize, const RANK: usize, D> ConvNd<SPATIAL, RANK, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Create a new convolution layer.
    ///
    /// `weight` shape: `(out_channels, in_channels / groups, ...kernel)`.
    /// `bias` shape: `(out_channels,)`.
    pub fn new(
        weight: Tensor<RANK, D, Concrete<D, RANK>>,
        bias: Option<Tensor<1, D, Concrete<D, 1>>>,
        config: ConvNdConfig<SPATIAL>,
    ) -> Self {
        // RANK = SPATIAL + 2 (batch + channels + spatial). Compile-time so a
        // misuse like `ConvNd::<2, 5, _>::new(...)` is rejected at the type
        // level rather than panicking at runtime.
        const {
            assert!(RANK == SPATIAL + 2);
        }
        let shape = weight.shape();
        let out_channels = shape[0];
        let in_channels = shape[1] * config.groups;

        assert!(config.groups >= 1, "groups must be >= 1");
        assert_eq!(
            out_channels % config.groups,
            0,
            "out_channels ({out_channels}) must be divisible by groups ({})",
            config.groups
        );

        if let Some(ref b) = bias {
            assert_eq!(
                b.shape()[0],
                out_channels,
                "Bias shape must match out_channels"
            );
        }

        Self {
            weight,
            bias,
            config,
            in_channels,
            out_channels,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &ConvNdConfig<SPATIAL> {
        &self.config
    }

    /// Number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

impl<const SPATIAL: usize, const RANK: usize, D> ConvNd<SPATIAL, RANK, D>
where
    D: SimdElement
        + DataType
        + FloatDataType
        + FloatOps
        + Default
        + MatmulImpl
        + std::ops::Mul<Output = D>
        + std::ops::Add<Output = D>,
{
    /// Forward pass for any spatial rank. The free const generic `R2` equals
    /// `RANK + SPATIAL` and is determined by the `LargerRank` bound, exactly
    /// the same way the underlying `composite::conv` operation infers it.
    pub fn forward<B, const R2: usize>(
        &self,
        input: &Tensor<RANK, D, B>,
    ) -> Tensor<RANK, D, Concrete<D, RANK>>
    where
        B: Fusion<RANK, D>,
        crate::MulOp: crate::cpu::SimdBinaryOp<D>,
        crate::AddOp: crate::cpu::SimdBinaryOp<D>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<D>,
        Concrete<D, RANK>: crate::cpu::LargerRank<R2, SPATIAL, D>,
        crate::gpu::Tensor<RANK, D>: crate::gpu::LargerRank<SPATIAL, R2, D>,
    {
        let input = input.to_concrete();

        if self.config.groups == 1 {
            input.conv(
                &self.weight,
                self.bias.as_ref(),
                self.config.padding,
                self.config.stride,
            )
        } else {
            input.grouped_conv(
                &self.weight,
                self.bias.as_ref(),
                self.config.padding,
                self.config.stride,
                self.config.groups,
            )
        }
    }
}

impl<const SPATIAL: usize, const RANK: usize> ConvNd<SPATIAL, RANK, f32> {
    /// Load from `VarBuilder`; bias is optional and loaded if present.
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: ConvNdConfig<SPATIAL>,
    ) -> crate::Result<Self> {
        let weight: Tensor<RANK, f32> = vb.get("weight", device)?.dequantize();
        let bias: Option<Tensor<1, f32, Concrete<f32, 1>>> =
            vb.get("bias", device).ok().map(|b| b.dequantize());

        Ok(Self::new(weight.to_concrete(), bias, config))
    }

    /// Load from `VarBuilder` without a bias term.
    pub fn load_no_bias(
        device: &Device,
        vb: &mut VarBuilder,
        config: ConvNdConfig<SPATIAL>,
    ) -> crate::Result<Self> {
        let weight: Tensor<RANK, f32> = vb.get("weight", device)?.dequantize();
        Ok(Self::new(weight.to_concrete(), None, config))
    }
}
