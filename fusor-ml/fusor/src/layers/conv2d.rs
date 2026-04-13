//! 2D Convolution layer implementation.

use crate::{ConcreteTensor, Device, MatmulImpl, SimdElement, Tensor, VarBuilder};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::{FloatOps, TensorBacking};

/// Configuration for Conv2d layer
#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    pub padding: [usize; 2],
    pub stride: [usize; 2],
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: [0, 0],
            stride: [1, 1],
            groups: 1,
        }
    }
}

/// 2D Convolution layer
///
/// Applies a 2D convolution over an input signal.
/// Input shape: (batch, in_channels, height, width)
/// Output shape: (batch, out_channels, out_height, out_width)
/// where out_h = (height + 2*padding[0] - kernel_h) / stride[0] + 1
///       out_w = (width + 2*padding[1] - kernel_w) / stride[1] + 1
pub struct Conv2d<D: SimdElement> {
    weight: Tensor<4, D, ConcreteTensor<D, 4>>,
    bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
    config: Conv2dConfig,
    in_channels: usize,
    out_channels: usize,
}

impl<D> Conv2d<D>
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
    /// Create a new Conv2d layer with given weights and configuration
    pub fn new(
        weight: Tensor<4, D, ConcreteTensor<D, 4>>,
        bias: Option<Tensor<1, D, ConcreteTensor<D, 1>>>,
        config: Conv2dConfig,
    ) -> Self {
        let shape = weight.shape();
        let out_channels = shape[0];
        let in_channels = shape[1] * config.groups;

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

    /// Forward pass
    ///
    /// Input shape: (batch, in_channels, height, width)
    /// Output shape: (batch, out_channels, out_height, out_width)
    pub fn forward(
        &self,
        input: &Tensor<4, D, impl TensorBacking<4, Elem = D>>,
    ) -> Tensor<4, D, ConcreteTensor<D, 4>>
    where
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
    {
        let input = input.to_concrete();
        if self.config.groups == 1 {
            input.conv(
                &self.weight,
                self.bias.as_ref(),
                self.config.padding,
                self.config.stride,
            )
        } else if self.config.groups == self.in_channels && self.in_channels == self.out_channels {
            // Depthwise convolution: each channel has its own filter
            // Fold channels into batch: (B, C, H, W) -> (B*C, 1, H, W)
            let shape = input.shape();
            let b = shape[0];
            let c = shape[1];
            let h = shape[2];
            let w = shape[3];

            let weight_shape = self.weight.shape();
            let kh = weight_shape[2];
            let kw = weight_shape[3];

            // Calculate output spatial dims
            let oh = (h + 2 * self.config.padding[0] - kh) / self.config.stride[0] + 1;
            let ow = (w + 2 * self.config.padding[1] - kw) / self.config.stride[1] + 1;

            // Use sliding_window_view approach directly:
            // 1. Pad input
            let padded: Tensor<4, D, ConcreteTensor<D, 4>> =
                if self.config.padding[0] > 0 || self.config.padding[1] > 0 {
                    let mut result: Tensor<4, D> = input.clone().into();
                    if self.config.padding[0] > 0 {
                        result = result.pad_axis(2, self.config.padding[0]);
                    }
                    if self.config.padding[1] > 0 {
                        result = result.pad_axis(3, self.config.padding[1]);
                    }
                    result.to_concrete()
                } else {
                    input.clone()
                };

            // 2. Create sliding windows: (B, C, oH, kH, oW, kW)
            use fusor_types::SlidingWindow;
            let windows: Tensor<6, D, ConcreteTensor<D, 6>> = padded
                .sliding_window_view([
                    SlidingWindow::new(2, kh, self.config.stride[0]),
                    SlidingWindow::new(3, kw, self.config.stride[1]),
                ])
                .to_concrete();

            // windows shape: (B, C, oH, oW, kH, kW)
            // Reshape weight from (C, 1, kH, kW) to (1, C, 1, 1, kH, kW) for broadcasting
            let weight_6d: Tensor<6, D> = self
                .weight
                .reshape([1, c, 1, 1, kh, kw])
                .broadcast_as([b, c, oh, ow, kh, kw])
                .to_concrete();

            // Element-wise multiply
            let product: Tensor<6, D> = (windows * weight_6d).to_concrete();

            // Sum over kernel dims (kH at axis 4, kW at axis 5)
            let sum_kw: Tensor<6, D> = product.sum_keepdim(5);
            let sum_khkw: Tensor<6, D> = sum_kw.sum_keepdim(4);
            // Shape is now (B, C, oH, oW, 1, 1) — squeeze the kernel dims
            let result: Tensor<4, D> = sum_khkw.reshape([b, c, oh, ow]).to_concrete();

            self.add_bias(result)
        } else {
            let g = self.config.groups;
            let in_ch_per_group = self.in_channels / g;
            let out_ch_per_group = self.out_channels / g;
            let mut outputs = Vec::with_capacity(g);
            for i in 0..g {
                let input_slice: Tensor<4, D, ConcreteTensor<D, 4>> = input
                    .narrow(1, i * in_ch_per_group, in_ch_per_group)
                    .to_concrete();
                let weight_slice: Tensor<4, D, ConcreteTensor<D, 4>> = self
                    .weight
                    .narrow(0, i * out_ch_per_group, out_ch_per_group)
                    .to_concrete();
                let group_out: Tensor<4, D> = input_slice.conv(
                    &weight_slice,
                    None::<&Tensor<1, D, ConcreteTensor<D, 1>>>,
                    self.config.padding,
                    self.config.stride,
                );
                outputs.push(group_out);
            }
            let result = Tensor::cat(outputs, 1);
            self.add_bias(result)
        }
    }

    /// Add bias to BCHW conv output.
    /// Bias shape is [out_channels], needs to be reshaped to [1, out_channels, 1, 1]
    /// for correct broadcasting along the channel dimension.
    fn add_bias<B: fusor_cpu::TensorBacking<4, Elem = D>>(
        &self,
        result: Tensor<4, D, B>,
    ) -> Tensor<4, D, ConcreteTensor<D, 4>>
    where
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
    {
        if let Some(bias) = &self.bias {
            let bias_4d: Tensor<4, D> = bias
                .reshape([1, self.out_channels, 1, 1])
                .broadcast_as(result.shape())
                .to_concrete();
            (result + bias_4d).to_concrete()
        } else {
            result.to_concrete()
        }
    }

    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

impl Conv2d<f32> {
    /// Load Conv2d layer from VarBuilder
    pub fn load(device: &Device, vb: &mut VarBuilder, config: Conv2dConfig) -> crate::Result<Self> {
        let weight: Tensor<4, f32> = vb.get("weight", device)?.dequantize();
        let bias: Option<Tensor<1, f32, ConcreteTensor<f32, 1>>> =
            vb.get("bias", device).ok().map(|b| b.dequantize());

        Ok(Self::new(weight.to_concrete(), bias, config))
    }

    /// Load Conv2d layer without bias from VarBuilder
    pub fn load_no_bias(
        device: &Device,
        vb: &mut VarBuilder,
        config: Conv2dConfig,
    ) -> crate::Result<Self> {
        let weight: Tensor<4, f32> = vb.get("weight", device)?.dequantize();
        Ok(Self::new(weight.to_concrete(), None, config))
    }
}
