use fusor::{
    cache::{AttentionMask, KvCache, MaskCache, TensorCache},
    layers::{
        BatchNorm1d, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear,
    },
    Device, Result, Tensor, VarBuilder,
};
use std::sync::Arc;

use crate::cohere_config::{CohereConfig, CohereDecoderConfig, CohereEncoderConfig};

fn conv_output_length(input: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    ((input + 2 * padding - kernel) / stride) + 1
}

fn rel_shift(x: &Tensor<4, f32>) -> Tensor<4, f32> {
    let [batch, heads, q_len, pos_len] = x.shape();
    let zeros = Tensor::zeros(&x.device(), [batch, heads, q_len, 1]);
    let padded = Tensor::cat([zeros, x.clone()], 3);
    let reshaped = padded
        .reshape([batch, heads, pos_len + 1, q_len])
        .to_concrete();
    reshaped
        .narrow(2, 1, pos_len)
        .reshape([batch, heads, q_len, pos_len])
        .to_concrete()
}

fn valid_mask(device: &Device, batch: usize, seq_len: usize, valid_len: usize) -> Tensor<2, f32> {
    let mut data = vec![0.0f32; batch * seq_len];
    for b in 0..batch {
        for i in 0..valid_len.min(seq_len) {
            data[b * seq_len + i] = 1.0;
        }
    }
    Tensor::from_slice(device, [batch, seq_len], &data)
}

fn encoder_attention_mask(
    device: &Device,
    batch: usize,
    seq_len: usize,
    valid_len: usize,
) -> Tensor<4, f32> {
    let mut data = vec![0.0f32; batch * seq_len * seq_len];
    for b in 0..batch {
        for q in 0..seq_len {
            for k in 0..seq_len {
                if q >= valid_len || k >= valid_len {
                    data[b * seq_len * seq_len + q * seq_len + k] = -1e9;
                }
            }
        }
    }
    Tensor::from_slice(device, [batch, 1, seq_len, seq_len], &data)
}

fn sigmoid(x: &Tensor<3, f32>) -> Tensor<3, f32> {
    ((-x).exp().add_scalar(1.0))
        .to_concrete()
        .div_scalar(1.0)
        .recip()
}

trait Reciprocal {
    fn recip(&self) -> Self;
}

impl Reciprocal for Tensor<3, f32> {
    fn recip(&self) -> Self {
        Tensor::splat(&self.device(), 1.0, self.shape())
            .div_(self)
            .to_concrete()
    }
}

struct ConvSubsampling {
    conv0: Conv2d<f32>,
    conv1_dw: Conv2d<f32>,
    conv1_pw: Conv2d<f32>,
    conv2_dw: Conv2d<f32>,
    conv2_pw: Conv2d<f32>,
    out: Linear<f32>,
}

impl ConvSubsampling {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: &CohereEncoderConfig) -> Result<Self> {
        let conv_stride = [2, 2];
        let conv_padding = [1, 1];
        let depthwise_cfg = Conv2dConfig {
            padding: conv_padding,
            stride: conv_stride,
            groups: cfg.subsampling_conv_channels,
            dilation: [1, 1],
        };
        let pointwise_cfg = Conv2dConfig {
            padding: [0, 0],
            stride: [1, 1],
            groups: 1,
            dilation: [1, 1],
        };
        Ok(Self {
            conv0: Conv2d::load(
                device,
                &mut vb.pp("conv.0"),
                Conv2dConfig {
                    padding: conv_padding,
                    stride: conv_stride,
                    groups: 1,
                    dilation: [1, 1],
                },
            )?,
            conv1_dw: Conv2d::load(device, &mut vb.pp("conv.2"), depthwise_cfg)?,
            conv1_pw: Conv2d::load(device, &mut vb.pp("conv.3"), pointwise_cfg)?,
            conv2_dw: Conv2d::load(device, &mut vb.pp("conv.5"), depthwise_cfg)?,
            conv2_pw: Conv2d::load(device, &mut vb.pp("conv.6"), pointwise_cfg)?,
            out: Linear::load(device, &mut vb.pp("out"))?,
        })
    }

    fn forward(&self, input: &Tensor<3, f32>, length: usize) -> (Tensor<3, f32>, usize) {
        let mut x = input.transpose(1, 2).unsqueeze(1).to_concrete();
        x = self.conv0.forward(&x).relu().to_concrete();
        x = self.conv1_dw.forward(&x).to_concrete();
        x = self.conv1_pw.forward(&x).relu().to_concrete();
        x = self.conv2_dw.forward(&x).to_concrete();
        x = self.conv2_pw.forward(&x).relu().to_concrete();

        let [batch, channels, time, freq] = x.shape();
        let x = x
            .transpose(1, 2)
            .reshape([batch, time, channels * freq])
            .to_concrete();
        let x = self.out.forward(&x);

        let mut reduced = length;
        for _ in 0..3 {
            reduced = conv_output_length(reduced, 3, 2, 1);
        }
        (x, reduced)
    }
}

struct RelPositionalEncoding {
    pe: Tensor<2, f32>,
}

impl RelPositionalEncoding {
    fn new(device: &Device, d_model: usize, max_len: usize) -> Self {
        let length = 2 * max_len - 1;
        let mut data = vec![0.0f32; length * d_model];
        let center = max_len as isize - 1;
        for pos_idx in 0..length {
            let position = (center - pos_idx as isize) as f32;
            for i in (0..d_model).step_by(2) {
                let div = (-(10000.0f32).ln() * i as f32 / d_model as f32).exp();
                data[pos_idx * d_model + i] = (position * div).sin();
                if i + 1 < d_model {
                    data[pos_idx * d_model + i + 1] = (position * div).cos();
                }
            }
        }
        Self {
            pe: Tensor::from_slice(device, [length, d_model], &data),
        }
    }

    fn forward(&self, x: &Tensor<3, f32>) -> (Tensor<3, f32>, Tensor<3, f32>) {
        let input_len = x.shape()[1];
        let center_pos = self.pe.shape()[0] / 2 + 1;
        let start_pos = center_pos - input_len;
        let pos_emb = self
            .pe
            .narrow(0, start_pos, 2 * input_len - 1)
            .unsqueeze(0)
            .to_concrete();
        (x.clone(), pos_emb)
    }
}

struct ConformerFeedForward {
    linear1: Linear<f32>,
    linear2: Linear<f32>,
}

impl ConformerFeedForward {
    fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: Linear::load(device, &mut vb.pp("linear1"))?,
            linear2: Linear::load(device, &mut vb.pp("linear2"))?,
        })
    }

    fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        self.linear2
            .forward(&self.linear1.forward(x).silu().to_concrete())
    }
}

struct ConformerConvolution {
    pointwise_conv1: Conv1d<f32>,
    depthwise_conv: Conv1d<f32>,
    batch_norm: BatchNorm1d<f32>,
    pointwise_conv2: Conv1d<f32>,
}

impl ConformerConvolution {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        d_model: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            pointwise_conv1: Conv1d::new(
                vb.pp("pointwise_conv1").get("weight", device)?.dequantize(),
                Some(vb.pp("pointwise_conv1").get("bias", device)?.dequantize()),
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    groups: 1,
                    dilation: 1,
                },
            ),
            depthwise_conv: Conv1d::new(
                vb.pp("depthwise_conv").get("weight", device)?.dequantize(),
                Some(vb.pp("depthwise_conv").get("bias", device)?.dequantize()),
                Conv1dConfig {
                    padding: (kernel_size - 1) / 2,
                    stride: 1,
                    groups: d_model,
                    dilation: 1,
                },
            ),
            batch_norm: BatchNorm1d::load(device, &mut vb.pp("batch_norm"), 1e-5)?,
            pointwise_conv2: Conv1d::new(
                vb.pp("pointwise_conv2").get("weight", device)?.dequantize(),
                Some(vb.pp("pointwise_conv2").get("bias", device)?.dequantize()),
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    groups: 1,
                    dilation: 1,
                },
            ),
        })
    }

    fn forward(&self, x: &Tensor<3, f32>, valid_mask: Option<&Tensor<2, f32>>) -> Tensor<3, f32> {
        let x = x.transpose(1, 2).to_concrete();
        let x = self.pointwise_conv1.forward(&x);
        let chunks = x.chunk(2, 1);
        let x = (chunks[0].to_concrete() * sigmoid(&chunks[1].to_concrete())).to_concrete();
        let x = if let Some(mask) = valid_mask {
            let mask_unsqueezed = mask.unsqueeze(1);
            let mask = mask_unsqueezed.broadcast_as(x.shape());
            (x * mask).to_concrete()
        } else {
            x
        };
        let x = self.depthwise_conv.forward(&x);
        let x = self.batch_norm.forward(&x).silu().to_concrete();
        self.pointwise_conv2
            .forward(&x)
            .transpose(1, 2)
            .to_concrete()
    }
}

struct RelPositionMultiHeadAttention {
    linear_q: Linear<f32>,
    linear_k: Linear<f32>,
    linear_v: Linear<f32>,
    linear_pos: Linear<f32>,
    linear_out: Linear<f32>,
    pos_bias_u: Tensor<2, f32>,
    pos_bias_v: Tensor<2, f32>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl RelPositionMultiHeadAttention {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        d_model: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        Ok(Self {
            linear_q: Linear::load(device, &mut vb.pp("linear_q"))?,
            linear_k: Linear::load(device, &mut vb.pp("linear_k"))?,
            linear_v: Linear::load(device, &mut vb.pp("linear_v"))?,
            linear_pos: Linear::load(device, &mut vb.pp("linear_pos"))?,
            linear_out: Linear::load(device, &mut vb.pp("linear_out"))?,
            pos_bias_u: vb.get("pos_bias_u", device)?.dequantize(),
            pos_bias_v: vb.get("pos_bias_v", device)?.dequantize(),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn reshape_heads(&self, x: &Tensor<3, f32>) -> Tensor<4, f32> {
        let [batch, time, _] = x.shape();
        x.reshape([batch, time, self.num_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete()
    }

    fn forward(
        &self,
        x: &Tensor<3, f32>,
        pos_emb: &Tensor<3, f32>,
        mask: Option<&Tensor<4, f32>>,
    ) -> Tensor<3, f32> {
        let [batch, time, _] = x.shape();
        let q = self.reshape_heads(&self.linear_q.forward(x));
        let k = self.reshape_heads(&self.linear_k.forward(x));
        let v = self.reshape_heads(&self.linear_v.forward(x));

        let pos = if pos_emb.shape()[0] == 1 && batch > 1 {
            pos_emb
                .broadcast_as([batch, pos_emb.shape()[1], pos_emb.shape()[2]])
                .to_concrete()
        } else {
            pos_emb.to_concrete()
        };
        let p = self
            .linear_pos
            .forward(&pos.to_concrete())
            .reshape([batch, pos.shape()[1], self.num_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete();

        let q_with_u = (q.clone()
            + self
                .pos_bias_u
                .reshape([1, self.num_heads, 1, self.head_dim])
                .broadcast_as(q.shape()))
        .to_concrete();
        let q_with_v = (q + self
            .pos_bias_v
            .reshape([1, self.num_heads, 1, self.head_dim])
            .broadcast_as([batch, self.num_heads, time, self.head_dim]))
        .to_concrete();

        let matrix_ac = q_with_u.mat_mul(&k.transpose(2, 3).to_concrete());
        let matrix_bd = rel_shift(&q_with_v.mat_mul(&p.transpose(2, 3).to_concrete()))
            .narrow(3, 0, matrix_ac.shape()[3])
            .to_concrete();
        let mut scores = (matrix_ac + matrix_bd).to_concrete().mul_scalar(self.scale);
        if let Some(mask) = mask {
            scores = scores.add_(mask);
        }
        let attn = scores.softmax_last_dim_fused();
        let context = attn
            .mat_mul(&v)
            .transpose(1, 2)
            .reshape([batch, time, self.num_heads * self.head_dim])
            .to_concrete();
        self.linear_out.forward(&context)
    }
}

struct ConformerLayer {
    norm_feed_forward1: LayerNorm<1, f32>,
    feed_forward1: ConformerFeedForward,
    norm_self_att: LayerNorm<1, f32>,
    self_attn: RelPositionMultiHeadAttention,
    norm_conv: LayerNorm<1, f32>,
    conv: ConformerConvolution,
    norm_feed_forward2: LayerNorm<1, f32>,
    feed_forward2: ConformerFeedForward,
    norm_out: LayerNorm<1, f32>,
}

impl ConformerLayer {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: &CohereEncoderConfig) -> Result<Self> {
        Ok(Self {
            norm_feed_forward1: LayerNorm::load(device, &mut vb.pp("norm_feed_forward1"), 1e-5)?,
            feed_forward1: ConformerFeedForward::load(device, &mut vb.pp("feed_forward1"))?,
            norm_self_att: LayerNorm::load(device, &mut vb.pp("norm_self_att"), 1e-5)?,
            self_attn: RelPositionMultiHeadAttention::load(
                device,
                &mut vb.pp("self_attn"),
                cfg.d_model,
                cfg.n_heads,
            )?,
            norm_conv: LayerNorm::load(device, &mut vb.pp("norm_conv"), 1e-5)?,
            conv: ConformerConvolution::load(
                device,
                &mut vb.pp("conv"),
                cfg.d_model,
                cfg.conv_kernel_size,
            )?,
            norm_feed_forward2: LayerNorm::load(device, &mut vb.pp("norm_feed_forward2"), 1e-5)?,
            feed_forward2: ConformerFeedForward::load(device, &mut vb.pp("feed_forward2"))?,
            norm_out: LayerNorm::load(device, &mut vb.pp("norm_out"), 1e-5)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor<3, f32>,
        pos_emb: &Tensor<3, f32>,
        mask: Option<&Tensor<4, f32>>,
        valid_mask: Option<&Tensor<2, f32>>,
    ) -> Tensor<3, f32> {
        let residual = x.clone();
        let x = (residual
            + self
                .feed_forward1
                .forward(&self.norm_feed_forward1.forward(x))
                .mul_scalar(0.5))
        .to_concrete();
        let residual = x.clone();
        let x = (residual
            + self
                .self_attn
                .forward(&self.norm_self_att.forward(&x), pos_emb, mask))
        .to_concrete();
        let residual = x.clone();
        let x =
            (residual + self.conv.forward(&self.norm_conv.forward(&x), valid_mask)).to_concrete();
        let residual = x.clone();
        self.norm_out.forward(
            &(residual
                + self
                    .feed_forward2
                    .forward(&self.norm_feed_forward2.forward(&x))
                    .mul_scalar(0.5))
            .to_concrete(),
        )
    }
}

struct ConformerEncoder {
    pre_encode: ConvSubsampling,
    pos_enc: RelPositionalEncoding,
    layers: Vec<ConformerLayer>,
}

impl ConformerEncoder {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: &CohereEncoderConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(ConformerLayer::load(
                device,
                &mut vb.pp(format!("layers.{i}")),
                cfg,
            )?);
        }
        Ok(Self {
            pre_encode: ConvSubsampling::load(device, &mut vb.pp("pre_encode"), cfg)?,
            pos_enc: RelPositionalEncoding::new(device, cfg.d_model, cfg.pos_emb_max_len),
            layers,
        })
    }

    fn forward(&self, input_features: &Tensor<3, f32>, length: usize) -> (Tensor<3, f32>, usize) {
        let (x, length) = self.pre_encode.forward(input_features, length);
        let time = x.shape()[1];
        let (mut x, pos_emb) = self.pos_enc.forward(&x);
        let valid = valid_mask(&x.device(), x.shape()[0], time, length);
        let att_mask = encoder_attention_mask(&x.device(), x.shape()[0], time, length);
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb, Some(&att_mask), Some(&valid));
        }
        (x, length)
    }
}

struct FixedPositionalEncoding {
    table: Embedding<f32>,
}

impl FixedPositionalEncoding {
    fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        Ok(Self {
            table: Embedding::new_from_tensor(vb.get("pos_enc", device)?.dequantize()),
        })
    }

    fn forward(&self, positions: &Tensor<2, u32>) -> Tensor<3, f32> {
        self.table.forward(positions)
    }
}

struct DecoderAttention {
    query_net: Linear<f32>,
    key_net: Linear<f32>,
    value_net: Linear<f32>,
    out_projection: Linear<f32>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

struct DecoderAttentionCache {
    kv_cache: KvCache<f32>,
}

impl DecoderAttentionCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            kv_cache: KvCache::new(1, max_seq_len),
        }
    }
}

impl DecoderAttention {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        hidden_size: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            query_net: Linear::load(device, &mut vb.pp("query_net"))?,
            key_net: Linear::load(device, &mut vb.pp("key_net"))?,
            value_net: Linear::load(device, &mut vb.pp("value_net"))?,
            out_projection: Linear::load(device, &mut vb.pp("out_projection"))?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn reshape_heads(&self, x: &Tensor<3, f32>) -> Tensor<4, f32> {
        let [batch, time, _] = x.shape();
        x.reshape([batch, time, self.num_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete()
    }

    fn forward_kv(
        &self,
        x: &Tensor<3, f32>,
        cache: Option<&mut DecoderAttentionCache>,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        let device = x.device();
        let key_states = self.key_net.forward(x);
        let value_states = self.value_net.forward(x);
        match cache {
            None => (key_states, value_states),
            Some(cache) => {
                let key_states_4d = key_states.unsqueeze(2).to_concrete();
                let value_states_4d = value_states.unsqueeze(2).to_concrete();
                let (k, v) = cache
                    .kv_cache
                    .append(&device, &key_states_4d, &value_states_4d);
                (k.squeeze(2).to_concrete(), v.squeeze(2).to_concrete())
            }
        }
    }

    fn qkv_attention(
        &self,
        q: &Tensor<3, f32>,
        k: &Tensor<3, f32>,
        v: &Tensor<3, f32>,
        attention_mask: Option<&AttentionMask<f32>>,
        attention_output: Option<&mut TensorCache<4, f32>>,
    ) -> Tensor<3, f32> {
        let [batch, q_time, _] = q.shape();
        let q = self.reshape_heads(q);
        let k = self.reshape_heads(k).transpose(2, 3).to_concrete();
        let v = self.reshape_heads(v);
        let mut scores = q.mat_mul(&k).to_concrete().mul_scalar(self.scale);
        if let Some(mask) = attention_mask {
            mask.forward(&mut scores);
        }
        if let Some(output) = attention_output {
            let last_query = scores.narrow(2, q_time - 1, 1).to_concrete();
            output.append(&q.device(), &last_query);
        }
        let attn = scores.softmax_last_dim_fused();
        let context = attn
            .mat_mul(&v)
            .transpose(1, 2)
            .reshape([batch, q_time, self.num_heads * self.head_dim])
            .to_concrete();
        self.out_projection.forward(&context)
    }

    fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        kv: (Tensor<3, f32>, Tensor<3, f32>),
        attention_mask: Option<&AttentionMask<f32>>,
        attention_output: Option<&mut TensorCache<4, f32>>,
    ) -> Tensor<3, f32> {
        let query_states = self.query_net.forward(hidden_states);
        let (key_states, value_states) = &kv;
        self.qkv_attention(
            &query_states,
            key_states,
            value_states,
            attention_mask,
            attention_output,
        )
    }
}

struct DecoderFeedForward {
    dense_in: Linear<f32>,
    dense_out: Linear<f32>,
    hidden_act: String,
}

impl DecoderFeedForward {
    fn load(device: &Device, vb: &mut VarBuilder, hidden_act: &str) -> Result<Self> {
        Ok(Self {
            dense_in: Linear::load(device, &mut vb.pp("dense_in"))?,
            dense_out: Linear::load(device, &mut vb.pp("dense_out"))?,
            hidden_act: hidden_act.to_owned(),
        })
    }

    fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        let x = self.dense_in.forward(x);
        let x = match self.hidden_act.as_str() {
            "relu" => x.relu().to_concrete(),
            "silu" | "swish" => x.silu().to_concrete(),
            _ => x.relu().to_concrete(),
        };
        self.dense_out.forward(&x)
    }
}

struct TransformerDecoderLayer {
    layer_norm_1: LayerNorm<1, f32>,
    first_sub_layer: DecoderAttention,
    layer_norm_2: LayerNorm<1, f32>,
    second_sub_layer: DecoderAttention,
    layer_norm_3: LayerNorm<1, f32>,
    third_sub_layer: DecoderFeedForward,
}

struct TransformerDecoderLayerCache {
    self_attn: DecoderAttentionCache,
    cross_attn_kv: (Tensor<3, f32>, Tensor<3, f32>),
}

impl TransformerDecoderLayer {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: &CohereDecoderConfig) -> Result<Self> {
        Ok(Self {
            layer_norm_1: LayerNorm::load(device, &mut vb.pp("layer_norm_1"), 1e-5)?,
            first_sub_layer: DecoderAttention::load(
                device,
                &mut vb.pp("first_sub_layer"),
                cfg.hidden_size,
                cfg.num_attention_heads,
            )?,
            layer_norm_2: LayerNorm::load(device, &mut vb.pp("layer_norm_2"), 1e-5)?,
            second_sub_layer: DecoderAttention::load(
                device,
                &mut vb.pp("second_sub_layer"),
                cfg.hidden_size,
                cfg.num_attention_heads,
            )?,
            layer_norm_3: LayerNorm::load(device, &mut vb.pp("layer_norm_3"), 1e-5)?,
            third_sub_layer: DecoderFeedForward::load(
                device,
                &mut vb.pp("third_sub_layer"),
                &cfg.hidden_act,
            )?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        encoder_hidden_states: &Tensor<3, f32>,
        self_attention_mask: &Tensor<4, f32>,
        _cross_attention_mask: Option<&Tensor<4, f32>>,
    ) -> Tensor<3, f32> {
        let residual = hidden_states.clone();
        let self_kv = self
            .first_sub_layer
            .forward_kv(&self.layer_norm_1.forward(hidden_states), None);
        let self_mask = AttentionMask::new(
            self_attention_mask
                .squeeze::<3>(0)
                .squeeze::<2>(0)
                .to_concrete(),
        );
        let hidden_states = (residual
            + self.first_sub_layer.forward(
                &self.layer_norm_1.forward(hidden_states),
                self_kv,
                Some(&self_mask),
                None,
            ))
        .to_concrete();
        let residual = hidden_states.clone();
        let cross_kv = self
            .second_sub_layer
            .forward_kv(encoder_hidden_states, None);
        let hidden_states = (residual
            + self.second_sub_layer.forward(
                &self.layer_norm_2.forward(&hidden_states),
                cross_kv,
                None,
                None,
            ))
        .to_concrete();
        let residual = hidden_states.clone();
        (residual
            + self
                .third_sub_layer
                .forward(&self.layer_norm_3.forward(&hidden_states)))
        .to_concrete()
    }

    fn forward_cached(
        &self,
        hidden_states: &Tensor<3, f32>,
        self_attention_mask: &AttentionMask<f32>,
        cache: &mut TransformerDecoderLayerCache,
        attention_output: Option<&mut TensorCache<4, f32>>,
    ) -> Tensor<3, f32> {
        let ln1 = self.layer_norm_1.forward(hidden_states);
        let self_kv = self
            .first_sub_layer
            .forward_kv(&ln1, Some(&mut cache.self_attn));
        let residual = hidden_states.clone();
        let hidden_states = (residual
            + self
                .first_sub_layer
                .forward(&ln1, self_kv, Some(self_attention_mask), None))
        .to_concrete();

        let residual = hidden_states.clone();
        let hidden_states = (residual
            + self.second_sub_layer.forward(
                &self.layer_norm_2.forward(&hidden_states),
                cache.cross_attn_kv.clone(),
                None,
                attention_output,
            ))
        .to_concrete();

        let residual = hidden_states.clone();
        (residual
            + self
                .third_sub_layer
                .forward(&self.layer_norm_3.forward(&hidden_states)))
        .to_concrete()
    }
}

struct TransformerDecoderEmbedding {
    token_embedding: Embedding<f32>,
    position_embedding: FixedPositionalEncoding,
    layer_norm: LayerNorm<1, f32>,
}

impl TransformerDecoderEmbedding {
    fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        Ok(Self {
            token_embedding: Embedding::load(device, &mut vb.pp("token_embedding"))?,
            position_embedding: FixedPositionalEncoding::load(
                device,
                &mut vb.pp("position_embedding"),
            )?,
            layer_norm: LayerNorm::load(device, &mut vb.pp("layer_norm"), 1e-5)?,
        })
    }

    fn forward(&self, input_ids: &Tensor<2, u32>, positions: &Tensor<2, u32>) -> Tensor<3, f32> {
        self.layer_norm.forward(
            &(self.token_embedding.forward(input_ids) + self.position_embedding.forward(positions))
                .to_concrete(),
        )
    }
}

struct TransformerDecoder {
    embedding: TransformerDecoderEmbedding,
    layers: Vec<TransformerDecoderLayer>,
    final_layer_norm: LayerNorm<1, f32>,
    max_target_positions: usize,
    mask_cache: Arc<MaskCache<f32>>,
}

#[derive(Default)]
struct TransformerDecoderCache {
    tokens: Vec<u32>,
    layers: Vec<TransformerDecoderLayerCache>,
}

impl TransformerDecoder {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: &CohereDecoderConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(TransformerDecoderLayer::load(
                device,
                &mut vb.pp(format!("_decoder.layers.{i}")),
                cfg,
            )?);
        }
        Ok(Self {
            embedding: TransformerDecoderEmbedding::load(device, &mut vb.pp("_embedding"))?,
            layers,
            final_layer_norm: LayerNorm::load(
                device,
                &mut vb.pp("_decoder.final_layer_norm"),
                1e-5,
            )?,
            max_target_positions: cfg.max_sequence_length,
            mask_cache: Default::default(),
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor<2, u32>,
        encoder_hidden_states: &Tensor<3, f32>,
        encoder_length: usize,
    ) -> Tensor<3, f32> {
        let [batch, seq_len] = input_ids.shape();
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let positions = Tensor::from_slice(&input_ids.device(), [batch, seq_len], &positions);
        let mut hidden_states = self.embedding.forward(input_ids, &positions);
        let self_mask = AttentionMask::causal(&input_ids.device(), seq_len);
        let cross_mask = if encoder_length < encoder_hidden_states.shape()[1] {
            Some(encoder_attention_mask(
                &input_ids.device(),
                encoder_hidden_states.shape()[0],
                encoder_hidden_states.shape()[1],
                encoder_length,
            ))
        } else {
            None
        };
        for layer in &self.layers {
            let self_mask_tensor = self_mask
                .mask()
                .clone()
                .unsqueeze(0)
                .unsqueeze(0)
                .to_concrete();
            hidden_states = layer.forward(
                &hidden_states,
                encoder_hidden_states,
                &self_mask_tensor,
                cross_mask.as_ref(),
            );
        }
        self.final_layer_norm.forward(&hidden_states)
    }

    fn forward_cached(
        &self,
        tokens: &[u32],
        encoder_hidden_states: &Tensor<3, f32>,
        _encoder_length: usize,
        cache: &mut TransformerDecoderCache,
        mut attention_output: Option<&mut [TensorCache<4, f32>]>,
    ) -> Tensor<3, f32> {
        let index_pos = cache.tokens.len();
        cache.tokens.extend_from_slice(tokens);
        let seq_len = tokens.len();
        assert!(
            index_pos + seq_len <= self.max_target_positions,
            "exceeded max sequence length"
        );

        let device = encoder_hidden_states.device();
        let self_mask = self.mask_cache.get_mask(seq_len, index_pos, None, &device);
        let token_tensor: Tensor<1, u32> = Tensor::new(&device, tokens);
        let token_tensor = token_tensor.unsqueeze(0).to_concrete();
        let positions: Vec<u32> = (index_pos as u32..(index_pos + seq_len) as u32).collect();
        let positions = Tensor::from_slice(&device, [1, seq_len], &positions);
        let mut hidden_states = self.embedding.forward(&token_tensor, &positions);

        for (i, layer) in self.layers.iter().enumerate() {
            if cache.layers.len() <= i {
                cache.layers.push(TransformerDecoderLayerCache {
                    self_attn: DecoderAttentionCache::new(self.max_target_positions),
                    cross_attn_kv: layer
                        .second_sub_layer
                        .forward_kv(encoder_hidden_states, None),
                });
            }
            let layer_cache = &mut cache.layers[i];
            let attention_output = attention_output.as_mut().map(|outputs| &mut outputs[i]);
            hidden_states =
                layer.forward_cached(&hidden_states, &self_mask, layer_cache, attention_output);
        }

        self.final_layer_norm.forward(&hidden_states)
    }
}

pub struct Cohere {
    pub config: CohereConfig,
    encoder: ConformerEncoder,
    decoder: TransformerDecoder,
    encoder_decoder_proj: Option<Linear<f32>>,
    lm_bias: Tensor<1, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cohere_audio::pcm_to_features;
    use std::path::Path;

    #[tokio::test]
    async fn debug_first_step_topk() {
        let root = Path::new("/tmp/cohere-transcribe-03-2026");
        if !root.exists() {
            return;
        }

        let config: CohereConfig =
            serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
        let weights = std::fs::read(root.join("model.gguf")).unwrap();
        let device = Device::cpu();
        let mut reader = std::io::Cursor::new(weights);
        let mut vb = VarBuilder::from_gguf(&mut reader).unwrap();
        let model = Cohere::load(&device, &mut vb, config.clone()).unwrap();

        let wav = hound::WavReader::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/samples_jfk.wav"
        ))
        .unwrap()
        .into_samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .take(16_000)
        .collect::<Vec<_>>();

        let filter_bytes = include_bytes!("../cohere_melfilters128.bytes").as_slice();
        let mut filterbank = vec![0.0f32; filter_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            filter_bytes,
            &mut filterbank,
        );
        for value in &mut filterbank {
            *value = half::bf16::from_f32(*value).to_f32();
        }

        let (features, total_frames, valid_frames) = pcm_to_features(&config, &wav, &filterbank);
        assert_eq!(total_frames, 101);
        assert_eq!(valid_frames, 100);
        let input_features = Tensor::from_slice(
            &device,
            [1, config.preprocessor.features, total_frames],
            &features,
        );
        let prompt_ids = [7_u32, 4, 16, 62, 62, 5, 9, 11, 13];

        let mut conv = input_features.transpose(1, 2).unsqueeze(1).to_concrete();
        conv = model
            .encoder
            .pre_encode
            .conv0
            .forward(&conv)
            .relu()
            .to_concrete();
        conv = model
            .encoder
            .pre_encode
            .conv1_dw
            .forward(&conv)
            .to_concrete();
        conv = model
            .encoder
            .pre_encode
            .conv1_pw
            .forward(&conv)
            .relu()
            .to_concrete();
        conv = model
            .encoder
            .pre_encode
            .conv2_dw
            .forward(&conv)
            .to_concrete();
        conv = model
            .encoder
            .pre_encode
            .conv2_pw
            .forward(&conv)
            .relu()
            .to_concrete();
        let conv_slice = conv.clone().as_slice().await.unwrap();
        let conv_shape = conv_slice.shape();
        let (conv_b, conv_c, conv_t, conv_f) =
            (conv_shape[0], conv_shape[1], conv_shape[2], conv_shape[3]);
        let mut conv_data = Vec::with_capacity(conv_b * conv_c * conv_t * conv_f);
        for b in 0..conv_b {
            for c in 0..conv_c {
                for t in 0..conv_t {
                    for f in 0..conv_f {
                        conv_data.push(conv_slice[[b, c, t, f]]);
                    }
                }
            }
        }
        let conv_sum: f32 = conv_data.iter().sum();
        let conv_mean = conv_sum / conv_data.len() as f32;
        let conv_var = conv_data
            .iter()
            .map(|value| {
                let diff = *value - conv_mean;
                diff * diff
            })
            .sum::<f32>()
            / conv_data.len() as f32;
        println!(
            "conv shape={:?} sum={} mean={} std={} first20={:?}",
            conv_slice.shape(),
            conv_sum,
            conv_mean,
            conv_var.sqrt(),
            &conv_data[..20]
        );

        let (pre_encode, pre_len) = model
            .encoder
            .pre_encode
            .forward(&input_features, valid_frames);
        let pre = pre_encode.clone().as_slice().await.unwrap();
        let pre_shape = pre.shape();
        let (pre_batch, pre_time, pre_hidden) = (pre_shape[0], pre_shape[1], pre_shape[2]);
        let mut pre_data = Vec::with_capacity(pre_batch * pre_time * pre_hidden);
        for b in 0..pre_batch {
            for t in 0..pre_time {
                for h in 0..pre_hidden {
                    pre_data.push(pre[[b, t, h]]);
                }
            }
        }
        let pre_sum: f32 = pre_data.iter().sum();
        let pre_mean = pre_sum / pre_data.len() as f32;
        let pre_var = pre_data
            .iter()
            .map(|value| {
                let diff = *value - pre_mean;
                diff * diff
            })
            .sum::<f32>()
            / pre_data.len() as f32;
        println!(
            "pre shape={:?} len={} sum={} mean={} std={} first20={:?}",
            pre.shape(),
            pre_len,
            pre_sum,
            pre_mean,
            pre_var.sqrt(),
            &pre_data[..20]
        );

        let (encoder_hidden_states, encoder_length) = model.encode(&input_features, valid_frames);
        let enc = encoder_hidden_states.clone().as_slice().await.unwrap();
        let shape = enc.shape();
        let (batch, time, hidden) = (shape[0], shape[1], shape[2]);
        let mut enc_data = Vec::with_capacity(batch * time * hidden);
        for b in 0..batch {
            for t in 0..time {
                for h in 0..hidden {
                    enc_data.push(enc[[b, t, h]]);
                }
            }
        }
        let enc_sum: f32 = enc_data.iter().sum();
        let enc_mean = enc_sum / enc_data.len() as f32;
        let enc_var = enc_data
            .iter()
            .map(|value| {
                let diff = *value - enc_mean;
                diff * diff
            })
            .sum::<f32>()
            / enc_data.len() as f32;
        println!(
            "enc shape={:?} len={} sum={} mean={} std={} first20={:?}",
            enc.shape(),
            encoder_length,
            enc_sum,
            enc_mean,
            enc_var.sqrt(),
            &enc_data[..20]
        );
        let input_ids = Tensor::from_slice(&device, [1, prompt_ids.len()], &prompt_ids);
        let hidden_states =
            model
                .decoder
                .forward(&input_ids, &encoder_hidden_states, encoder_length);
        let last_hidden = hidden_states
            .narrow(1, prompt_ids.len() - 1, 1)
            .to_concrete();
        let logits = model.lm_head(&last_hidden).as_slice().await.unwrap();

        let mut top = (0..config.vocab_size)
            .map(|token_id| (token_id, logits[[0, 0, token_id]]))
            .collect::<Vec<_>>();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("top10={:?}", &top[..10]);
    }
}

impl Cohere {
    pub fn load(device: &Device, vb: &mut VarBuilder, config: CohereConfig) -> Result<Self> {
        let encoder = ConformerEncoder::load(device, &mut vb.pp("encoder"), &config.encoder)?;
        let decoder = TransformerDecoder::load(
            device,
            &mut vb.pp("transf_decoder"),
            &config.transf_decoder.config_dict,
        )?;
        let encoder_decoder_proj =
            if config.encoder.d_model != config.transf_decoder.config_dict.hidden_size {
                Some(Linear::load(device, &mut vb.pp("encoder_decoder_proj"))?)
            } else {
                None
            };
        let lm_bias = vb
            .pp("log_softmax")
            .pp("mlp")
            .pp("layer0")
            .get("bias", device)?
            .dequantize();
        Ok(Self {
            config,
            encoder,
            decoder,
            encoder_decoder_proj,
            lm_bias,
        })
    }

    fn encode(&self, input_features: &Tensor<3, f32>, length: usize) -> (Tensor<3, f32>, usize) {
        let (encoder_hidden_states, encoder_length) = self.encoder.forward(input_features, length);
        if let Some(proj) = &self.encoder_decoder_proj {
            (proj.forward(&encoder_hidden_states), encoder_length)
        } else {
            (encoder_hidden_states, encoder_length)
        }
    }

    fn lm_head(&self, hidden_states: &Tensor<3, f32>) -> Tensor<3, f32> {
        hidden_states
            .q_mat_mul(
                self.decoder
                    .embedding
                    .token_embedding
                    .embeddings_quantized(),
            )
            .add_(&self.lm_bias)
    }

    pub async fn generate_greedy(
        &self,
        input_features: &Tensor<3, f32>,
        length: usize,
        prompt_ids: &[u32],
        eos_token_id: u32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let (encoder_hidden_states, encoder_length) = self.encode(input_features, length);
        let mut cache = TransformerDecoderCache::default();
        let mut tokens = prompt_ids.to_vec();
        let mut hidden_states = self.decoder.forward_cached(
            prompt_ids,
            &encoder_hidden_states,
            encoder_length,
            &mut cache,
            None,
        );

        for _ in 0..max_new_tokens {
            let last_hidden = hidden_states
                .narrow(1, hidden_states.shape()[1] - 1, 1)
                .to_concrete();
            let logits = self.lm_head(&last_hidden);
            let logits = logits.as_slice().await?;

            let mut best_token = 0u32;
            let mut best_value = f32::NEG_INFINITY;
            for token_id in 0..self.config.vocab_size {
                let value = logits[[0, 0, token_id]];
                if value > best_value {
                    best_value = value;
                    best_token = token_id as u32;
                }
            }

            if best_token == eos_token_id {
                break;
            }
            tokens.push(best_token);
            hidden_states = self.decoder.forward_cached(
                &[best_token],
                &encoder_hidden_states,
                encoder_length,
                &mut cache,
                None,
            );
        }

        Ok(tokens[prompt_ids.len()..].to_vec())
    }

    pub async fn generate_greedy_with_attention(
        &self,
        input_features: &Tensor<3, f32>,
        length: usize,
        prompt_ids: &[u32],
        eos_token_id: u32,
        max_new_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<Tensor<4, f32>>, usize)> {
        let (encoder_hidden_states, encoder_length) = self.encode(input_features, length);
        let mut tokens = prompt_ids.to_vec();
        let mut cache = TransformerDecoderCache::default();
        let mut attention_output: Vec<TensorCache<4, f32>> = (0..self.decoder.layers.len())
            .map(|_| TensorCache::new(2, max_new_tokens))
            .collect();
        let mut hidden_states = self.decoder.forward_cached(
            prompt_ids,
            &encoder_hidden_states,
            encoder_length,
            &mut cache,
            Some(&mut attention_output),
        );

        for _ in 0..max_new_tokens {
            let last_hidden = hidden_states
                .narrow(1, hidden_states.shape()[1] - 1, 1)
                .to_concrete();
            let logits = self.lm_head(&last_hidden);
            let logits = logits.as_slice().await?;

            let mut best_token = 0u32;
            let mut best_value = f32::NEG_INFINITY;
            for token_id in 0..self.config.vocab_size {
                let value = logits[[0, 0, token_id]];
                if value > best_value {
                    best_value = value;
                    best_token = token_id as u32;
                }
            }

            if best_token == eos_token_id {
                break;
            }
            tokens.push(best_token);
            hidden_states = self.decoder.forward_cached(
                &[best_token],
                &encoder_hidden_states,
                encoder_length,
                &mut cache,
                Some(&mut attention_output),
            );
        }

        let collected_attentions = attention_output
            .into_iter()
            .filter_map(|attentions| {
                if attentions.current_seq_len() == 0 {
                    None
                } else {
                    attentions.current_data().cloned()
                }
            })
            .collect();

        Ok((
            tokens[prompt_ids.len()..].to_vec(),
            collected_attentions,
            encoder_length,
        ))
    }
}
