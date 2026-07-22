//! Trainable N-dimensional convolution layer.

use crate::autograd::{AutogradElement, Graph, Tensor};

pub use crate::layers::ConvNdConfig;

/// N-dimensional convolution layer with trainable parameters.
///
/// Input / output tensors have rank `RANK = SPATIAL + 2`:
/// `(batch, channels, ...spatial)`.
/// Weight has shape `(out_channels, in_channels / groups, ...kernel)`.
pub struct ConvNd<const SPATIAL: usize, const RANK: usize, T: AutogradElement = f32> {
    weight: Tensor<RANK, T>,
    bias: Option<Tensor<1, T>>,
    config: ConvNdConfig<SPATIAL>,
    in_channels: usize,
    out_channels: usize,
}

impl<const SPATIAL: usize, const RANK: usize, T: AutogradElement> ConvNd<SPATIAL, RANK, T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
{
    /// Create a new convolution layer.
    ///
    /// `weight` shape: `(out_channels, in_channels / groups, ...kernel)`.
    /// `bias` shape: `(out_channels,)`.
    pub fn new(
        weight: Tensor<RANK, T>,
        bias: Option<Tensor<1, T>>,
        config: ConvNdConfig<SPATIAL>,
    ) -> Self {
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

    /// The weight leaf: `(out_channels, in_channels / groups, ...kernel)`.
    pub fn weight(&self) -> &Tensor<RANK, T> {
        &self.weight
    }

    /// The bias leaf: `(out_channels,)`.
    pub fn bias(&self) -> Option<&Tensor<1, T>> {
        self.bias.as_ref()
    }

    /// Forward pass for any spatial rank. The free const generic `R2` equals
    /// `RANK + SPATIAL` and is determined by the `LargerRank` bound, exactly
    /// the same way the underlying `conv` operation infers it.
    pub fn forward<const R2: usize>(&self, input: &Tensor<RANK, T>) -> Tensor<RANK, T>
    where
        crate::ConcreteTensor<T, RANK>: crate::cpu::LargerRank<R2, SPATIAL, T>,
        crate::gpu::Tensor<RANK, T>: crate::gpu::LargerRank<SPATIAL, R2, T>,
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

impl<const SPATIAL: usize, const RANK: usize> ConvNd<SPATIAL, RANK> {
    /// Import an inference layer's parameters as trainable leaves on `graph`.
    pub fn from_inference(
        graph: &Graph,
        layer: &crate::layers::ConvNd<SPATIAL, RANK, f32>,
    ) -> Self {
        Self::new(
            graph.leaf(layer.weight().clone()),
            layer.bias().map(|bias| graph.leaf(bias.clone())),
            *layer.config(),
        )
    }
}
