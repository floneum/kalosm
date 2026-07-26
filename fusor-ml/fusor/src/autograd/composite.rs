use crate::MaskKind;
use fusor_types::SlidingWindow;

use super::*;

impl<const R: usize, T: AutogradElement> Tensor<R, T>
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
    fn softmax_composite<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        let input_shape = self.shape();
        let max_values = self.max_keepdim_any::<OUT_RANK>(axis);
        let shifted = self.sub(&max_values.broadcast_as(input_shape));
        let exp_values = shifted.exp();
        let normalization = exp_values
            .sum_keepdim_any::<OUT_RANK>(axis)
            .broadcast_as(input_shape);
        exp_values.div(&normalization)
    }

    fn rms_norm_composite<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.layer_norm_composite::<W, OUT_RANK>(weight, bias, eps, false)
    }

    fn layer_norm_composite<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
        remove_mean: bool,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        let centered = if remove_mean {
            let mean = self.mean_keepdim_any::<OUT_RANK>(R - 1);
            self.sub(&mean.broadcast_as(self.shape()))
        } else {
            self.clone()
        };
        let variance = centered.sqr().mean_keepdim_any::<OUT_RANK>(R - 1);
        let std = variance.add_scalar(eps).sqrt();
        let normalized = centered.div(&std.broadcast_as(self.shape()));
        let scaled = normalized.mul(&weight.broadcast_as(self.shape()));
        if let Some(bias) = bias {
            scaled.add(&bias.broadcast_as(self.shape()))
        } else {
            scaled
        }
    }

    pub fn softmax<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        if axis == R - 1 {
            // Fused forward with composite replay backward: fewer kernels, same math.
            return self.softmax_last_dim_fused::<OUT_RANK>();
        }
        self.softmax_composite::<OUT_RANK>(axis)
    }

    pub fn softmax_last_dim<const OUT_RANK: usize>(&self) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.softmax::<OUT_RANK>(R - 1)
    }

    pub fn softmax_slow<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.softmax::<OUT_RANK>(axis)
    }

    pub fn softmax_slow_last_dim<const OUT_RANK: usize>(&self) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.softmax_slow::<OUT_RANK>(R - 1)
    }

    pub fn layer_norm<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<R, T>,
        bias: Option<&Tensor<R, T>>,
        eps: f32,
        remove_mean: bool,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.layer_norm_composite::<R, OUT_RANK>(weight, bias, eps, remove_mean)
    }

    pub fn rms_norm<const OUT_RANK: usize>(&self, weight: &Tensor<R, T>, eps: f32) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.rms_norm_composite::<R, OUT_RANK>(weight, None, eps)
    }

    pub fn softmax_last_dim_fused<const OUT_RANK: usize>(&self) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
    {
        let value = self.value.softmax_last_dim::<OUT_RANK>().into_concrete();
        // Analytic softmax backward: dS = P * (dP - rowsum(dP * P)).
        // The product inside the row sum is written inline so the reduce
        // absorbs it, and the output is a single P * (dP - s) expression.
        self.unary_from_value(value, move |grad, probs| {
            let shape = probs.shape();
            let row_sum = (&probs * &grad)
                .into_concrete()
                .sum_keepdim::<OUT_RANK>(R - 1);
            let shifted = (&grad - &row_sum.broadcast_as(shape)).into_concrete();
            (&probs * &shifted).into_concrete()
        })
    }

    pub fn rms_norm_fused<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
        crate::MulOp: crate::cpu::SimdBinaryOp<T>,
        crate::DivOp: crate::cpu::SimdBinaryOp<T>,
        crate::AddOp: crate::cpu::SimdBinaryOp<T>,
        crate::SqrtOp: crate::cpu::SimdUnaryOp<T>,
        (crate::gpu::Tensor<R, T>, crate::gpu::Tensor<W, T>): crate::gpu::MaxRank<R, T>,
        T: crate::CastTensor<f32>,
        f32: crate::CastTensor<T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
    {
        let value = self.value.rms_norm_fused::<W, OUT_RANK>(
            &weight.value,
            bias.as_ref().map(|bias| &bias.value),
            eps,
        );
        if let Some(bias) = bias {
            self.replay_ternary(
                weight,
                bias,
                "rms_norm_fused",
                value,
                move |input, weight, bias| {
                    input.rms_norm_composite::<W, OUT_RANK>(&weight, Some(&bias), eps)
                },
            )
        } else {
            self.replay_binary(weight, "rms_norm_fused", value, move |input, weight| {
                input.rms_norm_composite::<W, OUT_RANK>(&weight, None, eps)
            })
        }
    }

    pub fn rms_norm_fused_no_bias<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, T>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
        crate::MulOp: crate::cpu::SimdBinaryOp<T>,
        crate::DivOp: crate::cpu::SimdBinaryOp<T>,
        crate::AddOp: crate::cpu::SimdBinaryOp<T>,
        crate::SqrtOp: crate::cpu::SimdUnaryOp<T>,
        (crate::gpu::Tensor<R, T>, crate::gpu::Tensor<W, T>): crate::gpu::MaxRank<R, T>,
        T: crate::CastTensor<f32>,
        f32: crate::CastTensor<T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
    {
        self.rms_norm_fused::<W, OUT_RANK>(weight, None, eps)
    }

    pub fn rms_norm_residual_fused<const W: usize, const OUT_RANK: usize>(
        &self,
        residual: &Self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
        crate::MulOp: crate::cpu::SimdBinaryOp<T>,
        crate::DivOp: crate::cpu::SimdBinaryOp<T>,
        crate::AddOp: crate::cpu::SimdBinaryOp<T>,
        crate::SqrtOp: crate::cpu::SimdUnaryOp<T>,
        (crate::gpu::Tensor<R, T>, crate::gpu::Tensor<W, T>): crate::gpu::MaxRank<R, T>,
        T: crate::CastTensor<f32>,
        f32: crate::CastTensor<T>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
    {
        let value = self.value.rms_norm_residual_fused::<W, OUT_RANK, _>(
            &residual.value,
            &weight.value,
            bias.as_ref().map(|bias| &bias.value),
            eps,
        );
        match bias {
            None => self.replay_ternary(
                residual,
                weight,
                "rms_norm_residual_fused",
                value,
                move |input, residual, weight| {
                    input
                        .add(&residual)
                        .rms_norm_composite::<W, OUT_RANK>(&weight, None, eps)
                },
            ),
            Some(bias) => self.replay_quaternary(
                residual,
                weight,
                bias,
                "rms_norm_residual_fused",
                value,
                move |input, residual, weight, bias| {
                    input.add(&residual).rms_norm_composite::<W, OUT_RANK>(
                        &weight,
                        Some(&bias),
                        eps,
                    )
                },
            ),
        }
    }

    pub fn layer_norm_last_dim_fused<const OUT_RANK: usize, const W: usize>(
        &self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        <crate::gpu::Tensor<R, T> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, T>>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
        crate::AddOp: crate::cpu::SimdBinaryOp<T>,
        crate::SubOp: crate::cpu::SimdBinaryOp<T>,
        crate::MulOp: crate::cpu::SimdBinaryOp<T>,
        crate::DivOp: crate::cpu::SimdBinaryOp<T>,
        crate::SqrtOp: crate::cpu::SimdUnaryOp<T>,
    {
        let mut param_shape = [1usize; R];
        param_shape[R - 1] = self.shape()[R - 1];
        let weight_row = weight.value.reshape(param_shape);
        let weight_b = weight_row.broadcast_as(self.shape());
        let bias_row = bias.map(|bias| bias.value.reshape(param_shape));
        let bias_b = bias_row
            .as_ref()
            .map(|bias| bias.broadcast_as(self.shape()));
        let value = self.value.layer_norm::<OUT_RANK, _, _>(
            &weight_b,
            bias_b.as_ref(),
            T::from_f32(eps),
            true,
        );

        let input_id = self.handle.id;
        let weight_id = weight.handle.id;
        let bias_id = bias.map(|bias| bias.handle.id);
        let input_value = self.value.clone();
        let weight_value = weight.value.clone();
        let weight_shape = weight.value.shape();
        // Analytic layer-norm backward. Row statistics are recomputed here
        // instead of saved from the forward pass so the forward chain stays
        // exclusively consumed (and therefore fusable into one row program).
        let backward: BackwardRule = Arc::new(move |gradient| {
            let dy = downcast_tensor::<R, T>(&*gradient, "layer_norm_last_dim_fused")?;
            let shape = input_value.shape();
            let n = shape[R - 1];
            let rows: usize = shape.iter().take(R - 1).product();

            let x = input_value.to_concrete();
            let mean = x.mean_keepdim::<OUT_RANK>(R - 1);
            let centered = (&x - &mean.broadcast_as(shape)).into_concrete();
            let var = centered
                .sqr()
                .into_concrete()
                .mean_keepdim::<OUT_RANK>(R - 1);
            let std = var.add_scalar(T::from_f32(eps)).sqrt().into_concrete();
            let xhat = (&centered / &std.broadcast_as(shape)).into_concrete();

            let weight_row = weight_value.reshape(param_shape).into_concrete();
            let dxhat = (&dy * &weight_row.broadcast_as(shape)).into_concrete();
            let m1 = dxhat.mean_keepdim::<OUT_RANK>(R - 1);
            let dxhat_xhat = (&dxhat * &xhat).into_concrete();
            let m2 = dxhat_xhat.mean_keepdim::<OUT_RANK>(R - 1);
            let recentered = (&dxhat - &m1.broadcast_as(shape)).into_concrete();
            let projected = (&xhat * &m2.broadcast_as(shape)).into_concrete();
            let dx_num = (recentered - projected).into_concrete();
            let dx = (&dx_num / &std.broadcast_as(shape)).into_concrete();

            let dy_flat = dy.reshape([rows, n]).into_concrete();
            let xhat_flat = xhat.reshape([rows, n]).into_concrete();
            let dw_flat = (&dy_flat * &xhat_flat).into_concrete().sum::<1>(0);
            let dw = dw_flat.reshape(weight_shape).into_concrete();

            let mut targets = vec![
                BackwardTarget {
                    node: input_id,
                    gradient: Box::new(dx),
                },
                BackwardTarget {
                    node: weight_id,
                    gradient: Box::new(dw),
                },
            ];
            if let Some(bias_id) = bias_id {
                let db = dy_flat.sum::<1>(0).reshape(weight_shape).into_concrete();
                targets.push(BackwardTarget {
                    node: bias_id,
                    gradient: Box::new(db),
                });
            }
            Ok(targets)
        });
        let mut parents = vec![self.handle.clone(), weight.handle.clone()];
        if let Some(bias) = bias {
            parents.push(bias.handle.clone());
        }
        self.emit_op(value, parents, Some(backward))
    }

    fn pad_spatial<const DIFF: usize>(&self, padding: [usize; DIFF]) -> Self {
        let mut padded = self.clone();
        for (i, padding) in padding.into_iter().enumerate() {
            padded = padded.pad_axis(R - DIFF + i, padding);
        }
        padded
    }

    fn conv_output_shape<const DIFF: usize>(
        input_shape: [usize; R],
        out_channels: usize,
        kernel: [usize; DIFF],
        padding: [usize; DIFF],
        strides: [usize; DIFF],
    ) -> [usize; R] {
        let spatial_start = R - DIFF;
        let mut output_shape = input_shape;
        output_shape[1] = out_channels;
        for i in 0..DIFF {
            let padded_len = input_shape[spatial_start + i] + 2 * padding[i];
            output_shape[spatial_start + i] = (padded_len - kernel[i]) / strides[i] + 1;
        }
        output_shape
    }

    /// Pad + sliding-window view + flatten to one matmul row per output
    /// location: `(batch * out_spatial, in_channels * kernel_size)`.
    fn conv_windows_flat<const DIFF: usize, const R2: usize>(
        &self,
        kernel: [usize; DIFF],
        padding: [usize; DIFF],
        strides: [usize; DIFF],
    ) -> Tensor<2, T>
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LargerRank<R2, DIFF, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LargerRank<DIFF, R2, T>,
    {
        let input_shape = self.shape();
        let spatial_start = R - DIFF;
        let output_shape = Self::conv_output_shape(input_shape, 0, kernel, padding, strides);
        let windows: [SlidingWindow; DIFF] =
            std::array::from_fn(|i| SlidingWindow::new(spatial_start + i, kernel[i], strides[i]));
        let windows: Tensor<R2, T> = self.pad_spatial(padding).sliding_window_view(windows);
        let permutation: [usize; R2] = std::array::from_fn(|index| {
            if index == 0 {
                0
            } else if index <= DIFF {
                index + 1
            } else if index == DIFF + 1 {
                1
            } else {
                index
            }
        });
        let out_spatial_size: usize = output_shape[spatial_start..].iter().product();
        let kernel_size: usize = kernel.iter().product();
        windows.permute(permutation).reshape([
            input_shape[0] * out_spatial_size,
            input_shape[1] * kernel_size,
        ])
    }

    /// The same window operand as [`Self::conv_windows_flat`], built from one
    /// shifted slice per kernel offset instead of a sliding-window view, with
    /// columns ordered kernel-major (`offset * in_channels + channel`).
    ///
    /// A sliding-window view reads each input element from several output
    /// locations, so its transpose is an overlap-add — which the generic view
    /// backward expresses as a masked reduce over *every* (input position,
    /// output position) pair. That is quadratic in the spatial extent: a
    /// 768-long sequence spends 1.8 billion element visits per convolution to
    /// scatter 5.7 million gradients. Concatenated slices carry the same
    /// values with a linear backward — `cat` differentiates to slices and
    /// `narrow` to a zero-fill assign — and still materialize exactly once,
    /// because concatenating along the trailing (channel) axis of a
    /// channels-last view already produces the contiguous matmul operand.
    ///
    /// Only valid for unit strides; strided windows keep the view path.
    fn conv_windows_shifted<const DIFF: usize>(
        &self,
        kernel: [usize; DIFF],
        padding: [usize; DIFF],
        out_spatial: [usize; DIFF],
    ) -> Tensor<2, T> {
        let input_shape = self.shape();
        let in_channels = input_shape[1];
        let kernel_size: usize = kernel.iter().product();
        let out_spatial_size: usize = out_spatial.iter().product();
        // (batch, channels, ...spatial) -> (batch, ...spatial, channels)
        let channels_last: [usize; R] = std::array::from_fn(|axis| {
            if axis == 0 {
                0
            } else if axis < R - 1 {
                axis + 1
            } else {
                1
            }
        });
        let padded = self.pad_spatial(padding).permute(channels_last);
        let slices: Vec<Self> = (0..kernel_size)
            .map(|flat_offset| {
                let mut window = padded.clone();
                let mut rest = flat_offset;
                for axis in (0..DIFF).rev() {
                    let offset = rest % kernel[axis];
                    rest /= kernel[axis];
                    window = window.narrow(1 + axis, offset, out_spatial[axis]);
                }
                window
            })
            .collect();
        Self::cat(slices, R - 1).reshape([
            input_shape[0] * out_spatial_size,
            kernel_size * in_channels,
        ])
    }

    /// Reshape the `(batch * out_spatial, out_channels)` matmul output back to
    /// `(batch, out_channels, ...out_spatial)` and add the broadcast bias.
    fn conv_reassemble<const DIFF: usize>(
        output: Tensor<2, T>,
        bias: Option<&Tensor<1, T>>,
        output_shape: [usize; R],
    ) -> Self {
        let out_channels = output_shape[1];
        // Add the bias to the matmul's own `(rows, out_channels)` output,
        // where it broadcasts along the trailing axis, rather than after the
        // reassembly permute: an elementwise expression sitting directly on
        // the matmul result can ride its epilogue, while one behind a permute
        // costs a separate full pass over the activation.
        let output = match bias {
            Some(bias) => output.add_::<1, 2>(bias),
            None => output,
        };
        let output: Tensor<R, T> = output.reshape(std::array::from_fn(|axis| {
            if axis == 0 {
                output_shape[0]
            } else if axis <= DIFF {
                output_shape[axis + 1]
            } else {
                out_channels
            }
        }));
        let permutation: [usize; R] = std::array::from_fn(|index| {
            if index == 0 {
                0
            } else if index == 1 {
                DIFF + 1
            } else {
                index - 1
            }
        });
        output.permute(permutation)
    }

    fn conv_composite<const WEIGHT_RANK: usize, const DIFF: usize, const R2: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, T>,
        bias: Option<&Tensor<1, T>>,
        padding: [usize; DIFF],
        strides: [usize; DIFF],
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LargerRank<R2, DIFF, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LargerRank<DIFF, R2, T>,
    {
        assert_eq!(
            R,
            2 + DIFF,
            "Conv expects (batch, channels, ...spatial) format where R = 2 + DIFF"
        );
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let spatial_start = R - DIFF;
        let in_channels = input_shape[1];
        let out_channels = weight_shape[0];
        assert_eq!(
            weight_shape[1], in_channels,
            "Weight in_channels must match input in_channels"
        );

        let kernel: [usize; DIFF] = std::array::from_fn(|i| weight_shape[spatial_start + i]);
        let kernel_size: usize = kernel.iter().product();
        let output_shape =
            Self::conv_output_shape(input_shape, out_channels, kernel, padding, strides);

        // Unit strides take the shifted-slice operand, whose backward is
        // linear in the spatial extent; strided windows keep the view.
        // Its columns are kernel-major, so the weight is laid out to match:
        // (out, in, ...kernel) -> (...kernel, in, out).
        let output = if strides.iter().all(|stride| *stride == 1) {
            let out_spatial: [usize; DIFF] =
                std::array::from_fn(|i| output_shape[spatial_start + i]);
            let windows_flat = self.conv_windows_shifted::<DIFF>(kernel, padding, out_spatial);
            let kernel_major: [usize; WEIGHT_RANK] = std::array::from_fn(|axis| {
                if axis < DIFF {
                    axis + spatial_start
                } else if axis == DIFF {
                    1
                } else {
                    0
                }
            });
            let weight_rows = weight
                .permute(kernel_major)
                .reshape([kernel_size * in_channels, out_channels]);
            windows_flat.mat_mul_internal(&weight_rows)
        } else {
            let windows_flat = self.conv_windows_flat::<DIFF, R2>(kernel, padding, strides);
            let weight_t = weight
                .reshape([out_channels, in_channels * kernel_size])
                .transpose(0, 1);
            windows_flat.mat_mul_internal(&weight_t)
        };

        Self::conv_reassemble::<DIFF>(output, bias, output_shape)
    }

    pub fn conv<const WEIGHT_RANK: usize, const DIFF: usize, const R2: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, T>,
        bias: Option<&Tensor<1, T>>,
        padding: [usize; DIFF],
        strides: [usize; DIFF],
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LargerRank<R2, DIFF, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LargerRank<DIFF, R2, T>,
    {
        let value = self.value.conv::<WEIGHT_RANK, DIFF, R2>(
            &weight.value,
            bias.map(|bias| &bias.value),
            padding,
            strides,
        );
        match bias {
            None => self.replay_binary(weight, "conv", value, move |input, weight| {
                input.conv_composite::<WEIGHT_RANK, DIFF, R2>(&weight, None, padding, strides)
            }),
            Some(bias) => {
                self.replay_ternary(weight, bias, "conv", value, move |input, weight, bias| {
                    input.conv_composite::<WEIGHT_RANK, DIFF, R2>(
                        &weight,
                        Some(&bias),
                        padding,
                        strides,
                    )
                })
            }
        }
    }

    fn grouped_conv_composite<const WEIGHT_RANK: usize, const DIFF: usize, const R2: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, T>,
        bias: Option<&Tensor<1, T>>,
        padding: [usize; DIFF],
        strides: [usize; DIFF],
        groups: usize,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LargerRank<R2, DIFF, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LargerRank<DIFF, R2, T>,
    {
        assert_eq!(R, 2 + DIFF);
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let spatial_start = R - DIFF;
        let batch = input_shape[0];
        let in_channels = input_shape[1];
        let out_channels = weight_shape[0];
        assert_eq!(in_channels % groups, 0);
        assert_eq!(out_channels % groups, 0);
        let in_ch_per_group = in_channels / groups;
        let out_ch_per_group = out_channels / groups;
        assert_eq!(weight_shape[1], in_ch_per_group);

        let kernel: [usize; DIFF] = std::array::from_fn(|i| weight_shape[spatial_start + i]);
        let kernel_size: usize = kernel.iter().product();
        let output_shape =
            Self::conv_output_shape(input_shape, out_channels, kernel, padding, strides);
        let out_spatial_size: usize = output_shape[spatial_start..].iter().product();

        let windows_grouped = self
            .conv_windows_flat::<DIFF, R2>(kernel, padding, strides)
            .reshape([
                batch * out_spatial_size,
                groups,
                in_ch_per_group * kernel_size,
            ])
            .transpose(0, 1);
        let weight_grouped_t = weight
            .reshape([groups, out_ch_per_group, in_ch_per_group * kernel_size])
            .transpose(1, 2);
        let output = windows_grouped
            .mat_mul_internal(&weight_grouped_t)
            .transpose(0, 1)
            .reshape([batch * out_spatial_size, out_channels]);
        Self::conv_reassemble::<DIFF>(output, bias, output_shape)
    }

    pub fn grouped_conv<const WEIGHT_RANK: usize, const DIFF: usize, const R2: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, T>,
        bias: Option<&Tensor<1, T>>,
        padding: [usize; DIFF],
        strides: [usize; DIFF],
        groups: usize,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LargerRank<R2, DIFF, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LargerRank<DIFF, R2, T>,
    {
        let value = self.value.grouped_conv::<WEIGHT_RANK, DIFF, R2>(
            &weight.value,
            bias.map(|bias| &bias.value),
            padding,
            strides,
            groups,
        );
        match bias {
            None => self.replay_binary(weight, "grouped_conv", value, move |input, weight| {
                input.grouped_conv_composite::<WEIGHT_RANK, DIFF, R2>(
                    &weight, None, padding, strides, groups,
                )
            }),
            Some(bias) => self.replay_ternary(
                weight,
                bias,
                "grouped_conv",
                value,
                move |input, weight, bias| {
                    input.grouped_conv_composite::<WEIGHT_RANK, DIFF, R2>(
                        &weight,
                        Some(&bias),
                        padding,
                        strides,
                        groups,
                    )
                },
            ),
        }
    }
}

impl<T: AutogradElement> Tensor<4, T>
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
    fn rotate_half(&self) -> Tensor<4, T> {
        let [batch, heads, sequence_length, embed] = self.shape();
        let half = embed / 2;
        let first_half = self.narrow(3, 0, half);
        let second_half = self.narrow(3, half, embed - half).mul_scalar(-1.0);
        let graph = self.graph();
        let device = self.device();
        let zeros = Tensor::zeros(&graph, &device, [batch, heads, sequence_length, embed]);
        let combined = zeros.slice_assign(
            [0..batch, 0..heads, 0..sequence_length, 0..half],
            &second_half,
        );
        combined.slice_assign(
            [0..batch, 0..heads, 0..sequence_length, half..embed],
            &first_half,
        )
    }

    fn rope_interleaved_composite(&self, cos: &Tensor<2, T>, sin: &Tensor<2, T>) -> Tensor<4, T> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let [batch, heads, sequence_length, embed] = self.shape();
        let half = embed / 2;
        let cos = cos
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, half, 1])
            .broadcast_as([batch, heads, sequence_length, half, 1]);
        let sin = sin
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, half, 1])
            .broadcast_as([batch, heads, sequence_length, half, 1]);
        let x = self.reshape([batch, heads, sequence_length, half, 2]);
        let x0 = x.narrow(4, 0, 1);
        let x1 = x.narrow(4, 1, 1);
        let y0 = x0.mul(&cos).sub(&x1.mul(&sin));
        let y1 = x0.mul(&sin).add(&x1.mul(&cos));
        let graph = self.graph();
        let device = self.device();
        let zeros = Tensor::zeros(&graph, &device, [batch, heads, sequence_length, half, 2]);
        let combined =
            zeros.slice_assign([0..batch, 0..heads, 0..sequence_length, 0..half, 0..1], &y0);
        combined
            .slice_assign([0..batch, 0..heads, 0..sequence_length, 0..half, 1..2], &y1)
            .flatten_last_n::<1, 4>()
    }

    pub fn rope(&self, cos: &Tensor<2, T>, sin: &Tensor<2, T>) -> Tensor<4, T> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let [batch, heads, sequence_length, embed] = self.shape();
        let half = embed / 2;
        let graph = self.graph();
        let device = self.device();
        let cos_base = cos.narrow(0, 0, sequence_length);
        let sin_base = sin.narrow(0, 0, sequence_length);
        let cos = Tensor::zeros(&graph, &device, [sequence_length, embed])
            .slice_assign([0..sequence_length, 0..half], &cos_base)
            .slice_assign([0..sequence_length, half..embed], &cos_base)
            .unsqueeze_dims::<2, 4>([0, 1])
            .broadcast_as([batch, heads, sequence_length, embed]);
        let sin = Tensor::zeros(&graph, &device, [sequence_length, embed])
            .slice_assign([0..sequence_length, 0..half], &sin_base)
            .slice_assign([0..sequence_length, half..embed], &sin_base)
            .unsqueeze_dims::<2, 4>([0, 1])
            .broadcast_as([batch, heads, sequence_length, embed]);
        let rotated = self.rotate_half();
        self.mul(&cos).add(&rotated.mul(&sin))
    }

    pub fn rope_interleaved(&self, cos: &Tensor<2, T>, sin: &Tensor<2, T>) -> Tensor<4, T> {
        self.rope_interleaved_composite(cos, sin)
    }

    pub fn attention(
        &self,
        k: &Tensor<4, T>,
        v: &Tensor<4, T>,
        scale: f32,
        mask: Option<(&RawTensor<2, T>, MaskKind)>,
    ) -> Tensor<4, T> {
        let value = self.value.attention(&k.value, &v.value, scale, mask);
        let mask_value = mask.map(|(mask, kind)| (mask.clone(), kind));
        // The explicit rule recomputes probabilities from the forward output
        // and its row log-sum-exp, so no probability matrix survives into
        // the graph and pattern recognition can stream every piece.
        // Grouped-query, batch-key-masked, and CPU shapes replay the
        // composite as before.
        let explicit = self.shape()[1] == k.shape()[1]
            && matches!(&self.value, RawTensor::Gpu(_))
            && !matches!(mask, Some((_, MaskKind::BatchKeyMask)));
        if explicit {
            let q_id = self.handle.id;
            let k_id = k.handle.id;
            let v_id = v.handle.id;
            let q_value = self.value.clone();
            let k_value = k.value.clone();
            let v_value = v.value.clone();
            let o_value = value.clone();
            let backward: BackwardRule = Arc::new(move |gradient| {
                let grad = downcast_tensor::<4, T>(&*gradient, "attention")?;
                let (
                    RawTensor::Gpu(q),
                    RawTensor::Gpu(k),
                    RawTensor::Gpu(v),
                    RawTensor::Gpu(o),
                    RawTensor::Gpu(grad),
                ) = (&q_value, &k_value, &v_value, &o_value, &grad)
                else {
                    return Err(Error::msg("attention gradient expects GPU tensors"));
                };
                let causal = matches!(mask_value, Some((_, MaskKind::Causal)));
                let mask_gpu = match &mask_value {
                    Some((RawTensor::Gpu(mask), MaskKind::QKMask)) => Some(mask),
                    Some((_, MaskKind::QKMask)) => {
                        return Err(Error::msg("attention mask must be a GPU tensor"));
                    }
                    _ => None,
                };
                let lse = q.attention_lse(k, scale, mask_gpu, causal);
                let (dq, dk, dv) = q.attention_grads(k, v, o, grad, &lse, scale, mask_gpu, causal);
                Ok(vec![
                    BackwardTarget {
                        node: q_id,
                        gradient: Box::new(RawTensor::Gpu(dq)),
                    },
                    BackwardTarget {
                        node: k_id,
                        gradient: Box::new(RawTensor::Gpu(dk)),
                    },
                    BackwardTarget {
                        node: v_id,
                        gradient: Box::new(RawTensor::Gpu(dv)),
                    },
                ])
            });
            return self.emit_op(
                value,
                vec![self.handle.clone(), k.handle.clone(), v.handle.clone()],
                Some(backward),
            );
        }
        self.replay_ternary(k, v, "attention", value, move |q, k, v| {
            q.attention_composite(&k, &v, scale, mask_value.as_ref())
        })
    }

    pub fn rope_fused(&self, cos: &Tensor<2, T>, sin: &Tensor<2, T>) -> Tensor<4, T> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let value = self
            .value
            .rope_fused(&cos.value, &sin.value)
            .into_concrete();
        self.replay_ternary(cos, sin, "rope_fused", value, |input, cos, sin| {
            input.rope_interleaved_composite(&cos, &sin)
        })
    }

    pub fn rope_normal_fused(&self, cos: &Tensor<2, T>, sin: &Tensor<2, T>) -> Tensor<4, T> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let value = self
            .value
            .rope_normal_fused(&cos.value, &sin.value)
            .into_concrete();
        self.replay_ternary(cos, sin, "rope_normal_fused", value, |input, cos, sin| {
            input.rope(&cos, &sin)
        })
    }

    pub fn rope_pair_fused(
        &self,
        k: &Self,
        cos: &Tensor<2, T>,
        sin: &Tensor<2, T>,
    ) -> (Tensor<4, T>, Tensor<4, T>) {
        let (q_value, k_value) = self.value.rope_pair_fused(&k.value, &cos.value, &sin.value);
        (
            self.replay_ternary(
                cos,
                sin,
                "rope_pair_fused",
                q_value.into_concrete(),
                |input, cos, sin| input.rope_interleaved_composite(&cos, &sin),
            ),
            k.replay_ternary(
                cos,
                sin,
                "rope_pair_fused",
                k_value.into_concrete(),
                |input, cos, sin| input.rope_interleaved_composite(&cos, &sin),
            ),
        )
    }

    pub fn rope_normal_pair_fused(
        &self,
        k: &Self,
        cos: &Tensor<2, T>,
        sin: &Tensor<2, T>,
    ) -> (Tensor<4, T>, Tensor<4, T>) {
        let (q_value, k_value) = self
            .value
            .rope_normal_pair_fused(&k.value, &cos.value, &sin.value);
        (
            self.replay_ternary(
                cos,
                sin,
                "rope_normal_pair_fused",
                q_value.into_concrete(),
                |input, cos, sin| input.rope(&cos, &sin),
            ),
            k.replay_ternary(
                cos,
                sin,
                "rope_normal_pair_fused",
                k_value.into_concrete(),
                |input, cos, sin| input.rope(&cos, &sin),
            ),
        )
    }

    pub fn upsample_nearest2d(&self, scale_h: usize, scale_w: usize) -> Tensor<4, T> {
        let value = self.value.upsample_nearest2d(scale_h, scale_w);
        self.replay_unary("upsample_nearest2d", value, move |input| {
            let [b, c, h, w] = input.shape();
            input
                .reshape([b, c, h, 1, w, 1])
                .broadcast_as([b, c, h, scale_h, w, scale_w])
                .reshape([b, c, h * scale_h, w * scale_w])
        })
    }

    pub(super) fn attention_composite(
        &self,
        k: &Tensor<4, T>,
        v: &Tensor<4, T>,
        scale: f32,
        mask: Option<&(RawTensor<2, T>, MaskKind)>,
    ) -> Tensor<4, T> {
        let q_shape = self.shape();
        let k_shape = k.shape();
        let batch = q_shape[0];
        let num_heads = q_shape[1];
        let q_seq_len = q_shape[2];
        let head_dim = q_shape[3];
        let num_kv_heads = k_shape[1];
        let kv_seq_len = k_shape[2];
        assert!(
            num_heads.is_multiple_of(num_kv_heads),
            "Number of Q heads ({num_heads}) must be divisible by number of K/V heads ({num_kv_heads})"
        );

        let num_key_value_groups = num_heads / num_kv_heads;
        let (k_expanded, v_expanded) = if num_key_value_groups > 1 {
            let k_broadcast = k
                .reshape([batch, num_kv_heads, 1, kv_seq_len, head_dim])
                .broadcast_as([
                    batch,
                    num_kv_heads,
                    num_key_value_groups,
                    kv_seq_len,
                    head_dim,
                ]);
            let v_broadcast = v
                .reshape([batch, num_kv_heads, 1, kv_seq_len, head_dim])
                .broadcast_as([
                    batch,
                    num_kv_heads,
                    num_key_value_groups,
                    kv_seq_len,
                    head_dim,
                ]);
            (
                k_broadcast.reshape([batch, num_heads, kv_seq_len, head_dim]),
                v_broadcast.reshape([batch, num_heads, kv_seq_len, head_dim]),
            )
        } else {
            (k.clone(), v.clone())
        };

        let scores = self
            .mat_mul_internal(&k_expanded.transpose(2, 3))
            .div_scalar(scale.recip());
        let masked_scores = if let Some((mask, kind)) = mask {
            let mask_tensor = Tensor::constant_from_raw(&self.graph(), mask.clone());
            let mask_4d = match kind {
                // Causal falls back to QKMask semantics here: the provided mask is the
                // [seq_len, seq_len] (== [q_seq_len, kv_seq_len]) additive causal mask.
                MaskKind::QKMask | MaskKind::Causal => {
                    assert_eq!(mask_tensor.shape(), [q_seq_len, kv_seq_len]);
                    mask_tensor.reshape([1, 1, q_seq_len, kv_seq_len])
                }
                MaskKind::BatchKeyMask => {
                    assert_eq!(mask_tensor.shape(), [batch, kv_seq_len]);
                    mask_tensor.reshape([batch, 1, 1, kv_seq_len])
                }
            };
            scores.add(&mask_4d.broadcast_as([batch, num_heads, q_seq_len, kv_seq_len]))
        } else {
            scores
        };
        masked_scores
            .softmax_last_dim::<3>()
            .mat_mul_internal(&v_expanded)
    }
}

/// RoPE-cache companions stay on f32: [`crate::RopeCache`] holds f32 tables.
impl Tensor<4> {
    fn rope_cache_tables(
        &self,
        cache: &crate::RopeCache,
        start_pos: usize,
    ) -> (Tensor<2>, Tensor<2>) {
        let seq_len = self.shape()[2];
        let graph = self.graph();
        let table = |table: RawTensor<2, f32>| {
            Tensor::constant_from_raw(&graph, table.narrow(0, start_pos, seq_len).into_concrete())
        };
        (table(cache.cos().clone()), table(cache.sin().clone()))
    }

    /// Autograd companion of [`crate::RopeCache::forward`]: applies normal
    /// RoPE to `self` (q) and `k` from the cache's tables at `start_pos`.
    pub fn rope_cache_forward(
        &self,
        k: &Self,
        cache: &crate::RopeCache,
        start_pos: usize,
    ) -> (Tensor<4>, Tensor<4>) {
        let (cos, sin) = self.rope_cache_tables(cache, start_pos);
        self.rope_normal_pair_fused(k, &cos, &sin)
    }

    /// Autograd companion of [`crate::RopeCache::forward_interleaved`].
    pub fn rope_cache_forward_interleaved(
        &self,
        k: &Self,
        cache: &crate::RopeCache,
        start_pos: usize,
    ) -> (Tensor<4>, Tensor<4>) {
        let (cos, sin) = self.rope_cache_tables(cache, start_pos);
        self.rope_pair_fused(k, &cos, &sin)
    }
}
