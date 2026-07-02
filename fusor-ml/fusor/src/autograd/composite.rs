use crate::MaskKind;

use super::*;

impl<const R: usize> Tensor<R> {
    fn softmax_composite<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
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

    fn rms_norm_composite<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        bias: Option<&Tensor<1>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        let std = self
            .sqr()
            .mean_keepdim_any::<OUT_RANK>(R - 1)
            .add_scalar(eps)
            .sqrt();
        let normalized = self.div(&std.broadcast_as(self.shape()));
        let scaled = normalized.mul(&weight.broadcast_as(self.shape()));
        if let Some(bias) = bias {
            scaled.add(&bias.broadcast_as(self.shape()))
        } else {
            scaled
        }
    }

    fn layer_norm_composite<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        bias: Option<&Tensor<1>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        let centered = {
            let mean = self.mean_keepdim_any::<OUT_RANK>(R - 1);
            self.sub(&mean.broadcast_as(self.shape()))
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
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.softmax_composite::<OUT_RANK>(axis)
    }

    pub fn softmax_last_dim<const OUT_RANK: usize>(&self) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.softmax::<OUT_RANK>(R - 1)
    }

    pub fn softmax_last_dim_fused<const OUT_RANK: usize>(&self) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRankInner,
    {
        let value = self
            .value
            .softmax_last_dim_fused::<OUT_RANK>()
            .to_concrete();
        self.replay_unary("softmax_last_dim_fused", value, |input| {
            input.softmax_last_dim::<OUT_RANK>()
        })
    }

    pub fn rms_norm_fused<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        bias: Option<&Tensor<1>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, f32> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, f32>>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
        crate::MulOp: crate::cpu::SimdBinaryOp<f32>,
        crate::DivOp: crate::cpu::SimdBinaryOp<f32>,
        crate::AddOp: crate::cpu::SimdBinaryOp<f32>,
        crate::SqrtOp: crate::cpu::SimdUnaryOp<f32>,
        (crate::gpu::Tensor<R, f32>, crate::gpu::Tensor<1, f32>): crate::gpu::MaxRank<R, f32>,
    {
        let value = self.value.rms_norm_fused::<1, OUT_RANK>(
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
                    input.rms_norm_composite::<OUT_RANK>(&weight, Some(&bias), eps)
                },
            )
        } else {
            self.replay_binary(weight, "rms_norm_fused", value, move |input, weight| {
                input.rms_norm_composite::<OUT_RANK>(&weight, None, eps)
            })
        }
    }

    pub fn rms_norm_fused_no_bias<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, f32> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, f32>>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
        crate::MulOp: crate::cpu::SimdBinaryOp<f32>,
        crate::DivOp: crate::cpu::SimdBinaryOp<f32>,
        crate::AddOp: crate::cpu::SimdBinaryOp<f32>,
        crate::SqrtOp: crate::cpu::SimdUnaryOp<f32>,
        (crate::gpu::Tensor<R, f32>, crate::gpu::Tensor<1, f32>): crate::gpu::MaxRank<R, f32>,
    {
        self.rms_norm_fused::<OUT_RANK>(weight, None, eps)
    }

    pub fn layer_norm_last_dim_fused<const OUT_RANK: usize>(
        &self,
        weight: &Tensor<1>,
        bias: Option<&Tensor<1>>,
        eps: f32,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, f32> as crate::gpu::LastRankInner>::LastRank:
            crate::gpu::NextRankInner<NextRank = crate::gpu::Tensor<R, f32>>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
        crate::AddOp: crate::cpu::SimdBinaryOp<f32>,
        crate::SubOp: crate::cpu::SimdBinaryOp<f32>,
        crate::MulOp: crate::cpu::SimdBinaryOp<f32>,
        crate::DivOp: crate::cpu::SimdBinaryOp<f32>,
        crate::SqrtOp: crate::cpu::SimdUnaryOp<f32>,
    {
        let value = self.value.layer_norm_last_dim_fused::<OUT_RANK, 1, _, _>(
            &weight.value,
            bias.as_ref().map(|bias| &bias.value),
            eps,
        );
        if let Some(bias) = bias {
            self.replay_ternary(
                weight,
                bias,
                "layer_norm_last_dim_fused",
                value,
                move |input, weight, bias| {
                    input.layer_norm_composite::<OUT_RANK>(&weight, Some(&bias), eps)
                },
            )
        } else {
            self.replay_binary(
                weight,
                "layer_norm_last_dim_fused",
                value,
                move |input, weight| input.layer_norm_composite::<OUT_RANK>(&weight, None, eps),
            )
        }
    }
}

impl Tensor<2> {
    pub fn layer_norm(&self, weight: &Tensor<1>, bias: Option<&Tensor<1>>, eps: f32) -> Tensor<2> {
        self.layer_norm_composite::<1>(weight, bias, eps)
    }

    pub fn rms_norm(&self, weight: &Tensor<1>, eps: f32) -> Tensor<2> {
        self.rms_norm_composite::<1>(weight, None, eps)
    }
}

impl Tensor<3> {
    pub fn layer_norm(&self, weight: &Tensor<1>, bias: Option<&Tensor<1>>, eps: f32) -> Tensor<3> {
        self.layer_norm_composite::<2>(weight, bias, eps)
    }

    pub fn rms_norm(&self, weight: &Tensor<1>, eps: f32) -> Tensor<3> {
        self.rms_norm_composite::<2>(weight, None, eps)
    }
}

impl Tensor<4> {
    fn rotate_half(&self) -> Tensor<4> {
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

    fn rope_interleaved_composite(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let [batch, heads, sequence_length, embed] = self.shape();
        let half = embed / 2;
        let cos = cos
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, half, 1])
            .broadcast_as([batch, 1, sequence_length, half, 1]);
        let sin = sin
            .narrow(0, 0, sequence_length)
            .reshape([sequence_length, half, 1])
            .broadcast_as([batch, 1, sequence_length, half, 1]);
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

    pub fn rope(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
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

    pub fn rope_interleaved(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
        self.rope_interleaved_composite(cos, sin)
    }

    pub fn flash_attention(
        &self,
        k: &Tensor<4>,
        v: &Tensor<4>,
        scale: f32,
        mask: Option<(&RawTensor<2, f32>, MaskKind)>,
    ) -> Tensor<4> {
        let value = self.value.flash_attention(&k.value, &v.value, scale, mask);
        let mask_value = mask.map(|(mask, kind)| (mask.clone(), kind));
        self.replay_ternary(k, v, "flash_attention", value, move |q, k, v| {
            q.flash_attention_composite(&k, &v, scale, mask_value.as_ref())
        })
    }

    pub fn rope_fused(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let value = self.value.rope_fused(&cos.value, &sin.value).to_concrete();
        self.replay_ternary(cos, sin, "rope_fused", value, |input, cos, sin| {
            input.rope_interleaved_composite(&cos, &sin)
        })
    }

    pub fn rope_normal_fused(&self, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<4> {
        assert_same_graph(self, cos);
        assert_same_graph(self, sin);

        let value = self
            .value
            .rope_normal_fused(&cos.value, &sin.value)
            .to_concrete();
        self.replay_ternary(cos, sin, "rope_normal_fused", value, |input, cos, sin| {
            input.rope(&cos, &sin)
        })
    }

    pub fn layer_norm(&self, weight: &Tensor<1>, bias: Option<&Tensor<1>>, eps: f32) -> Tensor<4> {
        self.layer_norm_composite::<3>(weight, bias, eps)
    }

    pub fn rms_norm(&self, weight: &Tensor<1>, eps: f32) -> Tensor<4> {
        self.rms_norm_composite::<3>(weight, None, eps)
    }

    pub(super) fn flash_attention_composite(
        &self,
        k: &Tensor<4>,
        v: &Tensor<4>,
        scale: f32,
        mask: Option<&(RawTensor<2, f32>, MaskKind)>,
    ) -> Tensor<4> {
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
