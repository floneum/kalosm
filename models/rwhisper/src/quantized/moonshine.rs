use std::sync::Arc;

use fusor::{
    cache::{AttentionMask, KvCache, MaskCache, TensorCache},
    layers::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear},
    Device, Error, MaskKind, Result, RopeCache, Tensor, VarBuilder,
};

use crate::moonshine_config::{MoonshineStreamingConfig, MoonshineStreamingEncoderConfig};

fn tensor1d(device: &Device, vb: &mut VarBuilder, name: &str) -> Result<Tensor<1, f32>> {
    let q = vb.get(name, device)?;
    let shape = q.shape();
    Ok(if shape.len() == 1 {
        q.dequantize()
    } else {
        let value: Tensor<2, f32> = q.dequantize();
        if value.shape()[0] == 1 {
            value.squeeze(0).to_concrete()
        } else {
            value.squeeze(1).to_concrete()
        }
    })
}

fn conv1d(config: Conv1dConfig, device: &Device, vb: &mut VarBuilder) -> Result<Conv1d<f32>> {
    let weight_q = vb.get("weight", device)?;
    let weight_shape = weight_q.shape();
    let weight: Tensor<3, f32> = if weight_shape.len() == 3 {
        weight_q.dequantize()
    } else {
        let out = weight_shape[0];
        let kernel = 5usize;
        let in_channels = weight_shape[1] / kernel;
        let weight_2d: Tensor<2, f32> = weight_q.dequantize();
        weight_2d.reshape([out, in_channels, kernel]).to_concrete()
    };
    let bias = vb.get("bias", device).ok().map(|bias_q| {
        let bias_shape = bias_q.shape();
        if bias_shape.len() == 1 {
            bias_q.dequantize()
        } else {
            let bias_2d: Tensor<2, f32> = bias_q.dequantize();
            if bias_2d.shape()[0] == 1 {
                bias_2d.squeeze(0).to_concrete()
            } else {
                bias_2d.squeeze(1).to_concrete()
            }
        }
    });
    Ok(Conv1d::new(weight.to_concrete(), bias, config))
}

fn ones_1d(device: &Device, len: usize) -> Tensor<1, f32> {
    Tensor::splat(device, 1.0, [len])
}

fn causal_pad_1d(x: &Tensor<3, f32>, channels: usize, pad: usize) -> Tensor<3, f32> {
    let [batch, _, _] = x.shape();
    let zeros = Tensor::zeros(&x.device(), [batch, channels, pad]);
    Tensor::cat([zeros, x.clone()], 2)
}

fn flatten_frames(samples: &[f32], frame_len: usize, log_k: f32) -> Vec<f32> {
    let usable_len = samples.len() / frame_len * frame_len;
    let k = log_k.exp();
    let mut frames = Vec::with_capacity(usable_len);
    for frame in samples[..usable_len].chunks_exact(frame_len) {
        let mean = frame.iter().sum::<f32>() / frame_len as f32;
        let mut centered = vec![0.0f32; frame_len];
        let mut rms = 0.0f32;
        for (i, sample) in frame.iter().enumerate() {
            let value = *sample - mean;
            centered[i] = value;
            rms += value * value;
        }
        rms = (rms / frame_len as f32 + 1e-6).sqrt();
        for value in centered {
            let scaled = k * value / rms;
            frames.push((scaled + (scaled * scaled + 1.0).sqrt()).ln());
        }
    }
    frames
}

fn sliding_window_mask(
    device: &Device,
    seq_len: usize,
    left: usize,
    right: usize,
) -> Tensor<4, f32> {
    let mut data = vec![f32::NEG_INFINITY; seq_len * seq_len];
    for q in 0..seq_len {
        for k in 0..seq_len {
            let dist = q as isize - k as isize;
            let allowed =
                (dist >= 0 && dist < left as isize) || (dist < 0 && -dist < right as isize);
            if allowed {
                data[q * seq_len + k] = 0.0;
            }
        }
    }
    Tensor::from_slice(device, [1, 1, seq_len, seq_len], &data)
}

fn streaming_sliding_window_mask(
    device: &Device,
    query_start: usize,
    query_len: usize,
    key_len: usize,
    left: usize,
    right: usize,
) -> Tensor<4, f32> {
    let mut data = vec![f32::NEG_INFINITY; query_len * key_len];
    for q in 0..query_len {
        let query_index = query_start + q;
        for k in 0..key_len {
            let dist = query_index as isize - k as isize;
            let allowed =
                (dist >= 0 && dist < left as isize) || (dist < 0 && -dist < right as isize);
            if allowed {
                data[q * key_len + k] = 0.0;
            }
        }
    }
    Tensor::from_slice(device, [1, 1, query_len, key_len], &data)
}

fn causal_mask(device: &Device, seq_len: usize) -> Tensor<4, f32> {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for q in 0..seq_len {
        for k in (q + 1)..seq_len {
            data[q * seq_len + k] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_slice(device, [1, 1, seq_len, seq_len], &data)
}

fn stable_gelu_3d<B>(x: &Tensor<3, f32, B>) -> Tensor<3, f32>
where
    B: fusor::TensorBacking<3, Elem = f32>,
{
    let x = x.to_concrete();
    let x_squared = x.mul_(&x).to_concrete();
    let inner_factor = x_squared.mul_scalar(0.044715).add_scalar(1.0).to_concrete();
    let inner = x.mul_(&inner_factor).to_concrete();
    let tanh_input = inner
        .mul_scalar((2.0 / std::f32::consts::PI).sqrt())
        .to_concrete();
    let tanh_input = tanh_input.clamp(-10.0, 10.0).to_concrete();
    let tanh_result = tanh_input.tanh().clamp(-1.0, 1.0).to_concrete();
    let one_plus_tanh = tanh_result.add_scalar(1.0).to_concrete();
    x.mul_(&one_plus_tanh).mul_scalar(0.5).to_concrete()
}

struct UnitOffsetLayerNorm {
    norm: LayerNorm<1, f32>,
    gamma: Tensor<1, f32>,
    unit_offset: f32,
}

impl UnitOffsetLayerNorm {
    fn load(device: &Device, vb: &mut VarBuilder, eps: f32) -> Result<Self> {
        let gamma = tensor1d(device, vb, "gamma")?.to_concrete();
        let weight = ones_1d(device, gamma.shape()[0]).to_concrete();
        let norm = LayerNorm::new(weight, None, eps);
        Ok(Self {
            norm,
            gamma,
            unit_offset: 1.0,
        })
    }

    fn forward<B>(&self, x: &Tensor<3, f32, B>) -> Tensor<3, f32>
    where
        B: fusor::TensorBacking<3, Elem = f32>,
    {
        let normed = self.norm.forward(x);
        let gamma = self.gamma.add_scalar(self.unit_offset);
        let weight = gamma.broadcast_as(normed.shape());
        normed.mul_(&weight).to_concrete()
    }
}

#[derive(Clone)]
struct MoonshineAttentionCache {
    kv_cache: KvCache<f32>,
}

impl MoonshineAttentionCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            kv_cache: KvCache::new(1, max_seq_len),
        }
    }
}

struct MoonshineAttention {
    q_proj: Linear<f32>,
    k_proj: Linear<f32>,
    v_proj: Linear<f32>,
    o_proj: Linear<f32>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    rope_cache: Option<Arc<RopeCache>>,
    rotary_dim: usize,
}

impl MoonshineAttention {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        num_heads: usize,
        head_dim: usize,
        rope_cache: Option<Arc<RopeCache>>,
        rotary_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::load(device, &mut vb.pp("q_proj"))?,
            k_proj: Linear::load(device, &mut vb.pp("k_proj"))?,
            v_proj: Linear::load(device, &mut vb.pp("v_proj"))?,
            o_proj: Linear::load(device, &mut vb.pp("o_proj"))?,
            num_heads,
            head_dim,
            scale: head_dim as f32,
            rope_cache,
            rotary_dim,
        })
    }

    fn reshape_heads(&self, x: &Tensor<3, f32>) -> Tensor<4, f32> {
        let [batch, seq_len, _] = x.shape();
        x.reshape([batch, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete()
    }

    fn flatten_heads(&self, x: Tensor<4, f32>) -> Tensor<3, f32> {
        let [batch, _, seq_len, _] = x.shape();
        x.transpose(1, 2)
            .to_concrete()
            .reshape([batch, seq_len, self.num_heads * self.head_dim])
            .to_concrete()
    }

    fn apply_rope(
        &self,
        q: Tensor<4, f32>,
        k: Tensor<4, f32>,
        start_pos: usize,
    ) -> (Tensor<4, f32>, Tensor<4, f32>) {
        let Some(cache) = &self.rope_cache else {
            return (q, k);
        };
        if self.rotary_dim == 0 {
            return (q, k);
        }
        if self.rotary_dim < self.head_dim {
            return cache.forward_interleaved_partial(&q, &k, start_pos, self.rotary_dim);
        }
        let q_rot = q.narrow(3, 0, self.rotary_dim).to_concrete();
        let k_rot = k.narrow(3, 0, self.rotary_dim).to_concrete();
        let (q_rot, k_rot) = cache.forward_interleaved(&q_rot, &k_rot, start_pos);
        if self.rotary_dim == self.head_dim {
            return (q_rot, k_rot);
        }
        let q_pass = q
            .narrow(3, self.rotary_dim, self.head_dim - self.rotary_dim)
            .to_concrete();
        let k_pass = k
            .narrow(3, self.rotary_dim, self.head_dim - self.rotary_dim)
            .to_concrete();
        (
            Tensor::cat([q_rot, q_pass], 3),
            Tensor::cat([k_rot, k_pass], 3),
        )
    }

    fn apply_rope_3d(
        &self,
        q: Tensor<3, f32>,
        k: Tensor<3, f32>,
        start_pos: usize,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        if self.rope_cache.is_none() || self.rotary_dim == 0 {
            return (q, k);
        }
        let q = self.reshape_heads(&q);
        let k = self.reshape_heads(&k);
        let (q, k) = self.apply_rope(q, k, start_pos);
        (self.flatten_heads(q), self.flatten_heads(k))
    }

    fn project_kv(&self, hidden_states: &Tensor<3, f32>) -> (Tensor<3, f32>, Tensor<3, f32>) {
        (
            self.k_proj.forward(hidden_states),
            self.v_proj.forward(hidden_states),
        )
    }

    fn append_kv(
        &self,
        key_states: Tensor<3, f32>,
        value_states: Tensor<3, f32>,
        cache: Option<&mut MoonshineAttentionCache>,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        match cache {
            None => (key_states, value_states),
            Some(cache) => {
                let device = key_states.device();
                let key_states_4d = key_states.unsqueeze(2).to_concrete();
                let value_states_4d = value_states.unsqueeze(2).to_concrete();
                let (k, v) = cache
                    .kv_cache
                    .append(&device, &key_states_4d, &value_states_4d);
                (k.squeeze(2).to_concrete(), v.squeeze(2).to_concrete())
            }
        }
    }

    fn qkv_attention_dense(
        &self,
        q: &Tensor<3, f32>,
        k: &Tensor<3, f32>,
        v: &Tensor<3, f32>,
        attention_mask: Option<&Tensor<4, f32>>,
        attention_output: Option<&mut Vec<Tensor<4, f32>>>,
    ) -> Result<Tensor<3, f32>> {
        let [batch, q_len, _] = q.shape();
        let q = self.reshape_heads(q);
        let k = self.reshape_heads(k);
        let v = self.reshape_heads(v);
        let k_t = k.transpose(2, 3).to_concrete();
        let mut scores = q.mat_mul(&k_t).mul_scalar(self.scale.powf(-0.5));
        if let Some(mask) = attention_mask {
            scores = scores.add_(mask).to_concrete();
        }
        if let Some(outputs) = attention_output {
            outputs.push(scores.clone());
        }
        let scores = scores.to_concrete();
        let weights = scores.softmax_last_dim_fused().to_concrete();
        let context = weights
            .mat_mul(&v)
            .transpose(1, 2)
            .to_concrete()
            .reshape([batch, q_len, self.num_heads * self.head_dim])
            .to_concrete();
        Ok(self.o_proj.forward(&context))
    }

    fn qkv_attention_masked(
        &self,
        q: &Tensor<3, f32>,
        k: &Tensor<3, f32>,
        v: &Tensor<3, f32>,
        attention_mask: Option<&AttentionMask<f32>>,
        attention_output: Option<&mut TensorCache<4, f32>>,
    ) -> Result<Tensor<3, f32>> {
        let [batch, q_len, _] = q.shape();
        let q = self.reshape_heads(q);
        let k = self.reshape_heads(k);
        let v = self.reshape_heads(v);

        if attention_output.is_none() {
            let mask = attention_mask.map(|mask| (mask.mask(), MaskKind::QKMask));
            let context = q
                .flash_attention(&k, &v, self.scale.powf(-0.5), mask)
                .transpose(1, 2)
                .reshape([batch, q_len, self.num_heads * self.head_dim])
                .to_concrete();
            return Ok(self.o_proj.forward(&context));
        }

        let k_t = k.transpose(2, 3).to_concrete();
        let mut scores = q.mat_mul(&k_t).mul_scalar(self.scale.powf(-0.5));
        if q.is_gpu() {
            if let Some(mask) = attention_mask {
                let mask = mask.mask().clone().unsqueeze(0).unsqueeze(0).to_concrete();
                scores = scores.add_(&mask).to_concrete();
            }
        } else if let Some(mask) = attention_mask {
            mask.forward(&mut scores);
        }
        if let Some(output) = attention_output {
            let last_query = scores.narrow(2, q_len.saturating_sub(1), 1).to_concrete();
            output.append(&q.device(), &last_query);
        }
        let scores = scores.to_concrete();
        let weights = scores.softmax_last_dim_fused().to_concrete();
        let context = weights
            .mat_mul(&v)
            .transpose(1, 2)
            .to_concrete()
            .reshape([batch, q_len, self.num_heads * self.head_dim])
            .to_concrete();
        Ok(self.o_proj.forward(&context))
    }

    fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        key_value_states: &Tensor<3, f32>,
        attention_mask: Option<&Tensor<4, f32>>,
        attention_output: Option<&mut Vec<Tensor<4, f32>>>,
    ) -> Result<Tensor<3, f32>> {
        let query_states = self.q_proj.forward(hidden_states).to_concrete();
        let (key_states, value_states) = self.project_kv(key_value_states);
        let key_states = key_states.to_concrete();
        let value_states = value_states.to_concrete();
        let (query_states, key_states) = self.apply_rope_3d(query_states, key_states, 0);
        let query_states = query_states.to_concrete();
        let key_states = key_states.to_concrete();
        self.qkv_attention_dense(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
            attention_output,
        )
    }

    fn forward_cached(
        &self,
        hidden_states: &Tensor<3, f32>,
        kv: (Tensor<3, f32>, Tensor<3, f32>),
        attention_mask: Option<&AttentionMask<f32>>,
        attention_output: Option<&mut TensorCache<4, f32>>,
    ) -> Result<Tensor<3, f32>> {
        let query_states = self.q_proj.forward(hidden_states).to_concrete();
        let (key_states, value_states) = kv;
        let key_states = key_states.to_concrete();
        let value_states = value_states.to_concrete();
        self.qkv_attention_masked(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
            attention_output,
        )
    }
}

struct MoonshineEncoderMlp {
    fc1: Linear<f32>,
    fc2: Linear<f32>,
}

impl MoonshineEncoderMlp {
    fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(device, &mut vb.pp("fc1"))?,
            fc2: Linear::load(device, &mut vb.pp("fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        let hidden = self.fc1.forward(x).to_concrete();
        let hidden = stable_gelu_3d(&hidden);
        self.fc2.forward(&hidden)
    }
}

struct MoonshineDecoderMlp {
    fc1: Linear<f32>,
    fc2: Linear<f32>,
    intermediate_size: usize,
}

impl MoonshineDecoderMlp {
    fn load(device: &Device, vb: &mut VarBuilder, intermediate_size: usize) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(device, &mut vb.pp("fc1"))?,
            fc2: Linear::load(device, &mut vb.pp("fc2"))?,
            intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        let hidden = self.fc1.forward(x).to_concrete();
        self.fc2
            .forward(&hidden.swiglu_split(self.intermediate_size))
    }
}

struct MoonshineEncoderLayer {
    self_attn: MoonshineAttention,
    input_layernorm: UnitOffsetLayerNorm,
    post_attention_layernorm: UnitOffsetLayerNorm,
    mlp: MoonshineEncoderMlp,
}

#[derive(Clone, Default)]
pub struct MoonshineEncoderLayerStreamState {
    left_context_inputs: Option<Tensor<3, f32>>,
    pending_inputs: Option<Tensor<3, f32>>,
}

pub struct MoonshineEncoderStreamState {
    sample_remainder: Vec<f32>,
    linear_tail: Option<Tensor<3, f32>>,
    total_linear_frames: usize,
    conv1_tail: Option<Tensor<3, f32>>,
    total_conv1_frames: usize,
    layer_states: Vec<MoonshineEncoderLayerStreamState>,
    total_seen_frames: usize,
    total_finalized_frames: usize,
}

pub struct MoonshineEncoderStreamAppend {
    pub hidden_states: Option<Tensor<3, f32>>,
    pub total_seen_frames: usize,
    pub total_finalized_frames: usize,
    pub usable_input_samples: usize,
}

impl MoonshineEncoderLayer {
    fn load(
        config: &MoonshineStreamingEncoderConfig,
        device: &Device,
        vb: &mut VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: MoonshineAttention::load(
                device,
                &mut vb.pp("self_attn"),
                config.num_attention_heads,
                config.head_dim(),
                None,
                0,
            )?,
            input_layernorm: UnitOffsetLayerNorm::load(
                device,
                &mut vb.pp("input_layernorm"),
                1e-5,
            )?,
            post_attention_layernorm: UnitOffsetLayerNorm::load(
                device,
                &mut vb.pp("post_attention_layernorm"),
                1e-5,
            )?,
            mlp: MoonshineEncoderMlp::load(device, &mut vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        attention_mask: &Tensor<4, f32>,
    ) -> Result<Tensor<3, f32>> {
        let attn_in = self.input_layernorm.forward(hidden_states);
        let attn = self
            .self_attn
            .forward(&attn_in, &attn_in, Some(attention_mask), None)?;
        let hidden_states = (hidden_states + &attn).to_concrete();
        let mlp_in = self.post_attention_layernorm.forward(&hidden_states);
        let mlp = self.mlp.forward(&mlp_in);
        Ok((hidden_states + mlp).to_concrete())
    }

    fn forward_stream(
        &self,
        device: &Device,
        new_hidden_states: Option<Tensor<3, f32>>,
        state: &mut MoonshineEncoderLayerStreamState,
        left: usize,
        right: usize,
        flush: bool,
    ) -> Result<Option<Tensor<3, f32>>> {
        let left_context_len = state
            .left_context_inputs
            .as_ref()
            .map(|tensor| tensor.shape()[1])
            .unwrap_or(0);
        let pending_len = state
            .pending_inputs
            .as_ref()
            .map(|tensor| tensor.shape()[1])
            .unwrap_or(0);
        let new_len = new_hidden_states
            .as_ref()
            .map(|tensor| tensor.shape()[1])
            .unwrap_or(0);
        let working_len = pending_len + new_len;
        if left_context_len + working_len == 0 {
            return Ok(None);
        }

        let mut parts = Vec::with_capacity(3);
        if let Some(left_context_inputs) = &state.left_context_inputs {
            parts.push(left_context_inputs.clone());
        }
        if let Some(pending_inputs) = &state.pending_inputs {
            parts.push(pending_inputs.clone());
        }
        if let Some(new_hidden_states) = new_hidden_states {
            parts.push(new_hidden_states);
        }
        let combined_inputs = Tensor::cat(parts, 1).to_concrete();

        let emit_len = if flush {
            working_len
        } else {
            working_len.saturating_sub(right)
        };
        let pending_start = left_context_len + emit_len;
        let pending_len = working_len.saturating_sub(emit_len);
        state.pending_inputs = (pending_len > 0).then(|| {
            combined_inputs
                .narrow(1, pending_start, pending_len)
                .to_concrete()
        });

        let finalized_input_end = left_context_len + emit_len;
        let keep_left = left.min(finalized_input_end);
        state.left_context_inputs = (keep_left > 0).then(|| {
            combined_inputs
                .narrow(1, finalized_input_end - keep_left, keep_left)
                .to_concrete()
        });

        if emit_len == 0 {
            return Ok(None);
        }

        let key_value_inputs = self.input_layernorm.forward(&combined_inputs);
        let query_inputs = key_value_inputs
            .narrow(1, left_context_len, emit_len)
            .to_concrete();
        let residual_inputs = combined_inputs
            .narrow(1, left_context_len, emit_len)
            .to_concrete();
        let attention_mask = streaming_sliding_window_mask(
            device,
            left_context_len,
            emit_len,
            combined_inputs.shape()[1],
            left,
            right,
        );
        let attn = self.self_attn.forward(
            &query_inputs,
            &key_value_inputs,
            Some(&attention_mask),
            None,
        )?;
        let hidden_states = (residual_inputs + &attn).to_concrete();
        let mlp_in = self.post_attention_layernorm.forward(&hidden_states);
        let mlp = self.mlp.forward(&mlp_in);
        Ok(Some((hidden_states + mlp).to_concrete()))
    }
}

struct MoonshineDecoderLayer {
    self_attn: MoonshineAttention,
    encoder_attn: MoonshineAttention,
    input_layernorm: LayerNorm<1, f32>,
    post_attention_layernorm: LayerNorm<1, f32>,
    final_layernorm: LayerNorm<1, f32>,
    mlp: MoonshineDecoderMlp,
}

#[derive(Clone)]
struct MoonshineDecoderLayerCache {
    self_attn: MoonshineAttentionCache,
    cross_attn_kv: (Tensor<3, f32>, Tensor<3, f32>),
}

impl MoonshineDecoderLayer {
    fn load(
        config: &MoonshineStreamingConfig,
        device: &Device,
        vb: &mut VarBuilder,
        rope_cache: Arc<RopeCache>,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: MoonshineAttention::load(
                device,
                &mut vb.pp("self_attn"),
                config.num_attention_heads,
                config.decoder_head_dim(),
                Some(rope_cache),
                config.decoder_rotary_dim(),
            )?,
            encoder_attn: MoonshineAttention::load(
                device,
                &mut vb.pp("encoder_attn"),
                config.num_attention_heads,
                config.decoder_head_dim(),
                None,
                0,
            )?,
            input_layernorm: LayerNorm::load(device, &mut vb.pp("input_layernorm"), 1e-5)?,
            post_attention_layernorm: LayerNorm::load(
                device,
                &mut vb.pp("post_attention_layernorm"),
                1e-5,
            )?,
            final_layernorm: LayerNorm::load(device, &mut vb.pp("final_layernorm"), 1e-5)?,
            mlp: MoonshineDecoderMlp::load(device, &mut vb.pp("mlp"), config.intermediate_size)?,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor<3, f32>,
        encoder_hidden_states: &Tensor<3, f32>,
        causal_mask: &Tensor<4, f32>,
        attention_output: Option<&mut Vec<Tensor<4, f32>>>,
    ) -> Result<Tensor<3, f32>> {
        let self_attn_in = self.input_layernorm.forward_fused(hidden_states);
        let self_attn =
            self.self_attn
                .forward(&self_attn_in, &self_attn_in, Some(causal_mask), None)?;
        let hidden_states = (hidden_states + &self_attn).to_concrete();

        let cross_attn_in = self.post_attention_layernorm.forward_fused(&hidden_states);
        let cross_attn = self.encoder_attn.forward(
            &cross_attn_in,
            encoder_hidden_states,
            None,
            attention_output,
        )?;
        let hidden_states = (hidden_states + &cross_attn).to_concrete();

        let mlp_in = self.final_layernorm.forward_fused(&hidden_states);
        let mlp = self.mlp.forward(&mlp_in);
        Ok((hidden_states + mlp).to_concrete())
    }

    fn forward_cached(
        &mut self,
        hidden_states: &Tensor<3, f32>,
        self_attention_mask: &AttentionMask<f32>,
        index_pos: usize,
        cache: &mut MoonshineDecoderLayerCache,
        attention_output: Option<&mut TensorCache<4, f32>>,
    ) -> Result<Tensor<3, f32>> {
        let self_attn_in = self.input_layernorm.forward_fused(hidden_states);
        let query_states = self.self_attn.q_proj.forward(&self_attn_in);
        let (key_states, value_states) = self.self_attn.project_kv(&self_attn_in);
        let (query_states, key_states) =
            self.self_attn
                .apply_rope_3d(query_states, key_states, index_pos);
        let kv = self
            .self_attn
            .append_kv(key_states, value_states, Some(&mut cache.self_attn));
        let self_attn = self.self_attn.qkv_attention_masked(
            &query_states,
            &kv.0,
            &kv.1,
            Some(self_attention_mask),
            None,
        )?;
        let hidden_states = (hidden_states + &self_attn).to_concrete();

        let cross_attn_in = self.post_attention_layernorm.forward_fused(&hidden_states);
        let cross_attn = self.encoder_attn.forward_cached(
            &cross_attn_in,
            cache.cross_attn_kv.clone(),
            None,
            attention_output,
        )?;
        let hidden_states = (hidden_states + &cross_attn).to_concrete();

        let mlp_in = self.final_layernorm.forward_fused(&hidden_states);
        let mlp = self.mlp.forward(&mlp_in);
        Ok((hidden_states + mlp).to_concrete())
    }
}

pub struct MoonshineEncoder {
    linear: Linear<f32>,
    conv1: Conv1d<f32>,
    conv2: Conv1d<f32>,
    compression_log_k: f32,
    layers: Vec<MoonshineEncoderLayer>,
    final_norm: UnitOffsetLayerNorm,
    config: MoonshineStreamingEncoderConfig,
}

impl MoonshineEncoderStreamState {
    fn new(layer_count: usize) -> Self {
        Self {
            sample_remainder: Vec::new(),
            linear_tail: None,
            total_linear_frames: 0,
            conv1_tail: None,
            total_conv1_frames: 0,
            layer_states: (0..layer_count)
                .map(|_| MoonshineEncoderLayerStreamState::default())
                .collect(),
            total_seen_frames: 0,
            total_finalized_frames: 0,
        }
    }
}

impl MoonshineEncoder {
    fn load(
        config: &MoonshineStreamingEncoderConfig,
        device: &Device,
        vb: &mut VarBuilder,
    ) -> Result<Self> {
        let conv_config = Conv1dConfig {
            padding: 0,
            stride: 2,
            groups: 1,
            dilation: 1,
        };
        let layers = (0..config.num_hidden_layers)
            .map(|idx| {
                MoonshineEncoderLayer::load(config, device, &mut vb.pp(format!("layers.{idx}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let compression_log_k = {
            let log_k_q = vb.pp("embedder.comp").get("log_k", device)?;
            let log_k: Tensor<0, f32> = log_k_q.dequantize();
            pollster::block_on(log_k.to_scalar())?
        };
        Ok(Self {
            linear: Linear::load(device, &mut vb.pp("embedder.linear"))?,
            conv1: conv1d(conv_config, device, &mut vb.pp("embedder.conv1"))?,
            conv2: conv1d(conv_config, device, &mut vb.pp("embedder.conv2"))?,
            compression_log_k,
            layers,
            final_norm: UnitOffsetLayerNorm::load(device, &mut vb.pp("final_norm"), 1e-5)?,
            config: config.clone(),
        })
    }

    fn preprocess(&self, samples: &[f32]) -> Vec<f32> {
        flatten_frames(samples, self.config.frame_len(), self.compression_log_k)
    }

    pub fn new_stream_state(&self) -> MoonshineEncoderStreamState {
        MoonshineEncoderStreamState::new(self.layers.len())
    }

    fn ceil_div_2(value: usize) -> usize {
        (value + 1) / 2
    }

    fn take_last_len(
        tensor: &Tensor<3, f32>,
        dim: usize,
        max_len: usize,
    ) -> Option<Tensor<3, f32>> {
        let len = tensor.shape()[dim].min(max_len);
        (len > 0).then(|| {
            tensor
                .narrow(dim, tensor.shape()[dim] - len, len)
                .to_concrete()
        })
    }

    fn append_tail(
        previous_tail: Option<&Tensor<3, f32>>,
        new_values: &Tensor<3, f32>,
        dim: usize,
        max_len: usize,
    ) -> Option<Tensor<3, f32>> {
        let combined = match previous_tail {
            Some(previous_tail) => Tensor::cat([previous_tail.clone(), new_values.clone()], dim),
            None => new_values.clone(),
        };
        Self::take_last_len(&combined, dim, max_len)
    }

    fn append_conv1_outputs(
        &self,
        new_linear: Option<Tensor<3, f32>>,
        state: &mut MoonshineEncoderStreamState,
    ) -> Option<Tensor<3, f32>> {
        let Some(new_linear) = new_linear else {
            return None;
        };
        let old_linear_frames = state.total_linear_frames;
        let old_conv1_frames = state.total_conv1_frames;
        let new_linear_frames = new_linear.shape()[2];
        let total_linear_frames = old_linear_frames + new_linear_frames;
        let total_conv1_frames = Self::ceil_div_2(total_linear_frames);
        let new_conv1_frames = total_conv1_frames.saturating_sub(old_conv1_frames);

        let old_linear_tail = state.linear_tail.clone();
        state.linear_tail = Self::append_tail(state.linear_tail.as_ref(), &new_linear, 2, 4);
        state.total_linear_frames = total_linear_frames;
        state.total_conv1_frames = total_conv1_frames;

        if new_conv1_frames == 0 {
            return None;
        }

        let output = if old_linear_frames == 0 {
            self.conv1
                .forward(&causal_pad_1d(&new_linear, self.config.hidden_size, 4))
                .silu()
                .to_concrete()
        } else {
            let start_frame = old_conv1_frames.saturating_mul(2).saturating_sub(4);
            let prefix_len = old_linear_frames.saturating_sub(start_frame);
            let mut parts = Vec::with_capacity(2);
            if prefix_len > 0 {
                if let Some(tail) = &old_linear_tail {
                    parts.push(
                        tail.narrow(2, tail.shape()[2] - prefix_len, prefix_len)
                            .to_concrete(),
                    );
                }
            }
            parts.push(new_linear.clone());
            let window = Tensor::cat(parts, 2);
            let pad = self
                .conv1
                .kernel_size()
                .saturating_sub(1)
                .saturating_sub(old_conv1_frames.saturating_mul(self.conv1.config().stride));
            let window = if pad > 0 {
                causal_pad_1d(&window, self.config.hidden_size, pad)
            } else {
                window
            };
            self.conv1.forward(&window).silu().to_concrete()
        };

        Some(
            output
                .narrow(2, output.shape()[2] - new_conv1_frames, new_conv1_frames)
                .to_concrete(),
        )
    }

    fn append_encoder_outputs(
        &self,
        new_conv1: Option<Tensor<3, f32>>,
        state: &mut MoonshineEncoderStreamState,
    ) -> Option<Tensor<3, f32>> {
        let Some(new_conv1) = new_conv1 else {
            return None;
        };
        let old_conv1_frames = state
            .total_conv1_frames
            .saturating_sub(new_conv1.shape()[2]);
        let old_seen_frames = state.total_seen_frames;
        let total_conv1_frames = state.total_conv1_frames;
        let total_seen_frames = Self::ceil_div_2(total_conv1_frames);
        let new_seen_frames = total_seen_frames.saturating_sub(old_seen_frames);

        let old_conv1_tail = state.conv1_tail.clone();
        state.conv1_tail = Self::append_tail(state.conv1_tail.as_ref(), &new_conv1, 2, 4);
        state.total_seen_frames = total_seen_frames;

        if new_seen_frames == 0 {
            return None;
        }

        let output = if old_conv1_frames == 0 {
            self.conv2
                .forward(&causal_pad_1d(&new_conv1, self.config.hidden_size * 2, 4))
                .transpose(1, 2)
                .to_concrete()
        } else {
            let start_frame = old_seen_frames.saturating_mul(2).saturating_sub(4);
            let prefix_len = old_conv1_frames.saturating_sub(start_frame);
            let mut parts = Vec::with_capacity(2);
            if prefix_len > 0 {
                if let Some(tail) = &old_conv1_tail {
                    parts.push(
                        tail.narrow(2, tail.shape()[2] - prefix_len, prefix_len)
                            .to_concrete(),
                    );
                }
            }
            parts.push(new_conv1.clone());
            let window = Tensor::cat(parts, 2);
            let pad = self
                .conv2
                .kernel_size()
                .saturating_sub(1)
                .saturating_sub(old_seen_frames.saturating_mul(self.conv2.config().stride));
            let window = if pad > 0 {
                causal_pad_1d(&window, self.config.hidden_size * 2, pad)
            } else {
                window
            };
            self.conv2.forward(&window).transpose(1, 2).to_concrete()
        };

        Some(
            output
                .narrow(1, output.shape()[1] - new_seen_frames, new_seen_frames)
                .to_concrete(),
        )
    }

    pub fn encode_stream(
        &self,
        device: &Device,
        state: &mut MoonshineEncoderStreamState,
        samples: &[f32],
        flush: bool,
    ) -> Result<MoonshineEncoderStreamAppend> {
        if !samples.is_empty() {
            state.sample_remainder.extend_from_slice(samples);
        }
        let frame_len = self.config.frame_len();
        let usable_sample_count = state.sample_remainder.len() / frame_len * frame_len;
        let new_hidden_states = if usable_sample_count > 0 {
            let frames = self.preprocess(&state.sample_remainder[..usable_sample_count]);
            state.sample_remainder.drain(..usable_sample_count);
            let frame_count = frames.len() / frame_len;
            let hidden_states = Tensor::from_slice(device, [1, frame_count, frame_len], &frames);
            let hidden_states = self.linear.forward(&hidden_states).silu().to_concrete();
            Some(hidden_states.transpose(1, 2).to_concrete())
        } else {
            None
        };

        let mut hidden_states = self.append_conv1_outputs(new_hidden_states, state);
        hidden_states = self.append_encoder_outputs(hidden_states, state);

        for (idx, layer) in self.layers.iter().enumerate() {
            let window = self.config.sliding_windows[idx];
            hidden_states = layer.forward_stream(
                device,
                hidden_states,
                &mut state.layer_states[idx],
                window[0],
                window[1],
                flush,
            )?;
        }

        let finalized_hidden_states = hidden_states
            .map(|hidden_states| self.final_norm.forward(&hidden_states).to_concrete());
        if let Some(finalized_hidden_states) = &finalized_hidden_states {
            state.total_finalized_frames += finalized_hidden_states.shape()[1];
        }

        Ok(MoonshineEncoderStreamAppend {
            hidden_states: finalized_hidden_states,
            total_seen_frames: state.total_seen_frames,
            total_finalized_frames: state.total_finalized_frames,
            usable_input_samples: state.total_linear_frames * frame_len,
        })
    }

    pub fn encode(&self, device: &Device, samples: &[f32]) -> Result<Tensor<3, f32>> {
        let frames = self.preprocess(samples);
        let frame_len = self.config.frame_len();
        if frames.is_empty() {
            return Err(Error::msg("moonshine input is too short"));
        }
        let frame_count = frames.len() / frame_len;
        let mut hidden_states = Tensor::from_slice(device, [1, frame_count, frame_len], &frames);
        hidden_states = self.linear.forward(&hidden_states).silu();
        let mut hidden_states = hidden_states.transpose(1, 2).to_concrete();
        hidden_states = self
            .conv1
            .forward(&causal_pad_1d(&hidden_states, self.config.hidden_size, 4))
            .silu()
            .to_concrete();
        hidden_states = self
            .conv2
            .forward(&causal_pad_1d(
                &hidden_states,
                self.config.hidden_size * 2,
                4,
            ))
            .transpose(1, 2)
            .to_concrete();

        let seq_len = hidden_states.shape()[1];
        for (idx, layer) in self.layers.iter().enumerate() {
            let window = self.config.sliding_windows[idx];
            let mask = sliding_window_mask(device, seq_len, window[0], window[1]);
            hidden_states = layer.forward(&hidden_states, &mask)?;
        }

        Ok(self.final_norm.forward(&hidden_states))
    }
}

pub struct MoonshineDecoder {
    token_embedding: Embedding<f32>,
    pos_embedding: Embedding<f32>,
    proj: Option<Linear<f32>>,
    layers: Vec<MoonshineDecoderLayer>,
    norm: LayerNorm<1, f32>,
    output: Linear<f32>,
    max_target_positions: usize,
    mask_cache: Arc<MaskCache<f32>>,
    config: MoonshineStreamingConfig,
}

#[derive(Clone, Default)]
pub struct MoonshineDecoderCache {
    pub(crate) tokens: Vec<u32>,
    layers: Vec<MoonshineDecoderLayerCache>,
    pub(crate) encoder_seq_len: usize,
}

impl MoonshineDecoder {
    fn load(
        config: &MoonshineStreamingConfig,
        device: &Device,
        vb: &mut VarBuilder,
    ) -> Result<Self> {
        let rope_cache = Arc::new(RopeCache::new(
            config.decoder_rotary_dim(),
            config.max_position_embeddings,
            config.rope_theta(),
            device,
        )?);
        let layers = (0..config.num_hidden_layers)
            .map(|idx| {
                MoonshineDecoderLayer::load(
                    config,
                    device,
                    &mut vb.pp(format!("layers.{idx}")),
                    rope_cache.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let proj = (config.encoder_hidden_size() != config.hidden_size)
            .then(|| Linear::load(device, &mut vb.pp("proj")))
            .transpose()?;
        Ok(Self {
            token_embedding: Embedding::load(device, &mut vb.pp("embed_tokens"))?,
            pos_embedding: Embedding::load(device, &mut vb.pp("pos_emb"))?,
            proj,
            layers,
            norm: LayerNorm::load(device, &mut vb.pp("norm"), 1e-5)?,
            output: Linear::load(device, &mut vb.pp("proj_out"))?,
            max_target_positions: config.max_position_embeddings,
            mask_cache: Default::default(),
            config: config.clone(),
        })
    }

    fn adapt_encoder(
        &self,
        encoder_hidden_states: &Tensor<3, f32>,
        start_pos: usize,
    ) -> Result<Tensor<3, f32>> {
        let [_, seq_len, _] = encoder_hidden_states.shape();
        if start_pos + seq_len > self.config.max_position_embeddings {
            return Err(Error::msg(
                "moonshine encoder sequence exceeds max_position_embeddings",
            ));
        }
        let positions: Vec<u32> = (start_pos as u32..(start_pos + seq_len) as u32).collect();
        let position_ids: Tensor<1, u32> = Tensor::new(&encoder_hidden_states.device(), &positions);
        let position_embeddings: Tensor<2, f32> = self.pos_embedding.forward(&position_ids);
        let position_embeddings = position_embeddings.unsqueeze(0).to_concrete();
        let hidden_states = (encoder_hidden_states + &position_embeddings).to_concrete();
        Ok(match &self.proj {
            Some(proj) => proj.forward(&hidden_states),
            None => hidden_states,
        })
    }

    pub fn prepare_encoder_hidden_states(
        &self,
        encoder_hidden_states: &Tensor<3, f32>,
    ) -> Result<Tensor<3, f32>> {
        self.prepare_encoder_hidden_states_range(encoder_hidden_states, 0)
    }

    pub fn prepare_encoder_hidden_states_range(
        &self,
        encoder_hidden_states: &Tensor<3, f32>,
        start_pos: usize,
    ) -> Result<Tensor<3, f32>> {
        Ok(self
            .adapt_encoder(encoder_hidden_states, start_pos)?
            .to_concrete())
    }

    pub fn decode_prepared(
        &mut self,
        tokens: &[u32],
        encoder_hidden_states: &Tensor<3, f32>,
        attention_output: Option<&mut Vec<Tensor<4, f32>>>,
    ) -> Result<Tensor<3, f32>> {
        let device = encoder_hidden_states.device();
        let token_ids: Tensor<2, u32> = Tensor::new(&device, tokens)
            .reshape([1, tokens.len()])
            .to_concrete();
        let mut hidden_states: Tensor<3, f32> = self.token_embedding.forward(&token_ids);
        let mask = causal_mask(&device, tokens.len());

        let mut attention_output = attention_output;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(
                &hidden_states,
                encoder_hidden_states,
                &mask,
                attention_output.as_mut().map(|outputs| &mut **outputs),
            )?;
        }
        let hidden_states = self.norm.forward_fused(&hidden_states);
        Ok(self.output.forward(&hidden_states))
    }

    pub fn decode_cached(
        &mut self,
        tokens: &[u32],
        encoder_hidden_states: &Tensor<3, f32>,
        cache: &mut MoonshineDecoderCache,
    ) -> Result<Tensor<3, f32>> {
        let index_pos = cache.tokens.len();
        let seq_len = tokens.len();
        if index_pos + seq_len > self.max_target_positions {
            return Err(Error::msg(
                "moonshine decoder sequence exceeds max_position_embeddings",
            ));
        }
        cache.tokens.extend_from_slice(tokens);
        self.sync_cross_attention_kv(encoder_hidden_states, cache)?;

        let device = encoder_hidden_states.device();
        let self_mask = self.mask_cache.get_mask(seq_len, index_pos, None, &device);
        let token_tensor: Tensor<1, u32> = Tensor::new(&device, tokens);
        let token_tensor = token_tensor.unsqueeze(0).to_concrete();
        let mut hidden_states: Tensor<3, f32> = self.token_embedding.forward(&token_tensor);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            if cache.layers.len() <= i {
                let cross_attn_kv = layer.encoder_attn.project_kv(encoder_hidden_states);
                cache.layers.push(MoonshineDecoderLayerCache {
                    self_attn: MoonshineAttentionCache::new(self.max_target_positions),
                    cross_attn_kv,
                });
            }

            hidden_states = layer.forward_cached(
                &hidden_states,
                &self_mask,
                index_pos,
                &mut cache.layers[i],
                None,
            )?;
        }

        cache.encoder_seq_len = encoder_hidden_states.shape()[1];
        let hidden_states = self.norm.forward_fused(&hidden_states);
        Ok(self.output.forward(&hidden_states))
    }

    pub fn prepare_cross_attention_cache(
        &mut self,
        encoder_hidden_states: &Tensor<3, f32>,
        cache: &mut MoonshineDecoderCache,
    ) -> Result<()> {
        self.sync_cross_attention_kv(encoder_hidden_states, cache)
    }

    pub fn decode(
        &mut self,
        tokens: &[u32],
        encoder_hidden_states: &Tensor<3, f32>,
        attention_output: Option<&mut Vec<Tensor<4, f32>>>,
    ) -> Result<Tensor<3, f32>> {
        let encoder_hidden_states = self.prepare_encoder_hidden_states(encoder_hidden_states)?;
        self.decode_prepared(tokens, &encoder_hidden_states, attention_output)
    }

    fn sync_cross_attention_kv(
        &mut self,
        encoder_hidden_states: &Tensor<3, f32>,
        cache: &mut MoonshineDecoderCache,
    ) -> Result<()> {
        let encoder_seq_len = encoder_hidden_states.shape()[1];
        if encoder_seq_len == 0 {
            cache.encoder_seq_len = 0;
            return Ok(());
        }
        if cache.layers.is_empty() {
            for layer in self.layers.iter_mut() {
                let cross_attn_kv = layer.encoder_attn.project_kv(encoder_hidden_states);
                cache.layers.push(MoonshineDecoderLayerCache {
                    self_attn: MoonshineAttentionCache::new(self.max_target_positions),
                    cross_attn_kv,
                });
            }
            cache.encoder_seq_len = encoder_seq_len;
            return Ok(());
        }
        if cache.encoder_seq_len >= encoder_seq_len {
            return Ok(());
        }

        let new_encoder_hidden_states = encoder_hidden_states
            .narrow(
                1,
                cache.encoder_seq_len,
                encoder_seq_len - cache.encoder_seq_len,
            )
            .to_concrete();
        for (layer, layer_cache) in self.layers.iter_mut().zip(cache.layers.iter_mut()) {
            let new_cross_attn_kv = layer.encoder_attn.project_kv(&new_encoder_hidden_states);
            layer_cache.cross_attn_kv = (
                Tensor::cat(
                    [layer_cache.cross_attn_kv.0.clone(), new_cross_attn_kv.0],
                    1,
                )
                .to_concrete(),
                Tensor::cat(
                    [layer_cache.cross_attn_kv.1.clone(), new_cross_attn_kv.1],
                    1,
                )
                .to_concrete(),
            );
        }
        cache.encoder_seq_len = encoder_seq_len;
        Ok(())
    }
}

pub struct Moonshine {
    pub encoder: MoonshineEncoder,
    pub decoder: MoonshineDecoder,
}

impl Moonshine {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: MoonshineStreamingConfig,
    ) -> Result<Self> {
        let encoder =
            MoonshineEncoder::load(&config.encoder_config, device, &mut vb.pp("model.encoder"))?;
        let decoder = MoonshineDecoder::load(&config, device, &mut vb.pp("model.decoder"))?;
        Ok(Self { encoder, decoder })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moonshine_config::MoonshineStreamingConfig;
    use std::{fs, io::Cursor, path::Path, path::PathBuf};

    fn load_tiny_from_single_gguf() -> Option<(Moonshine, MoonshineStreamingConfig, Device)> {
        load_tiny_with_device(Device::cpu())
    }

    fn load_tiny_with_device(
        device: Device,
    ) -> Option<(Moonshine, MoonshineStreamingConfig, Device)> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts")
            .join("moonshine-streaming-tiny.gguf");
        let path = if path.exists() {
            path
        } else if let Ok(path) = std::env::var("RWHISPER_MOONSHINE_GGUF") {
            PathBuf::from(path)
        } else {
            path
        };
        if !path.exists() {
            return None;
        }
        let weights = fs::read(&path).unwrap();
        let mut reader = Cursor::new(weights);
        let mut vb = VarBuilder::from_gguf(&mut reader).unwrap();
        let config_json = vb
            .get_metadata("rwhisper.config.json")
            .expect("missing rwhisper.config.json metadata in GGUF")
            .to_string()
            .unwrap();
        let config: MoonshineStreamingConfig = serde_json::from_str(&config_json).unwrap();
        let model = Moonshine::load(&device, &mut vb, config.clone()).unwrap();
        Some((model, config, device))
    }

    fn read_jfk_pcm(target_rate: usize) -> Vec<f32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/samples_jfk.wav");
        let bytes = fs::read(&path).unwrap();
        let decoder = rodio::Decoder::new(std::io::Cursor::new(bytes)).unwrap();
        let resampled = rodio::source::UniformSourceIterator::new(decoder, 1, target_rate as u32);
        resampled.map(|s: i16| s as f32 / i16::MAX as f32).collect()
    }

    // Regression guard for the streaming encoder: chunked encode + flush must
    // match the non-streaming encoder on real audio. Broken previously because
    // `append_conv1_outputs` / `append_encoder_outputs` updated the conv tail
    // before reading it for the next window, so later chunks padded their conv
    // input with already-current frames instead of the prior chunk's tail.
    #[tokio::test]
    async fn streaming_encoder_matches_full_encoder_on_jfk() {
        let Some((model, config, device)) = load_tiny_from_single_gguf() else {
            return;
        };

        let frame_len = config.encoder_config.frame_len();
        let sample_rate = config.encoder_config.sample_rate;
        let raw = read_jfk_pcm(sample_rate);
        let usable_len = raw.len() / frame_len * frame_len;
        let samples = raw[..usable_len].to_vec();

        let full = model.encoder.encode(&device, &samples).unwrap();
        let full = full.as_slice().await.unwrap();

        let chunk_samples = samples.len() / 2 / frame_len * frame_len;
        let mut stream_state = model.encoder.new_stream_state();
        let mut streamed_parts = Vec::new();
        for chunk in samples.chunks(chunk_samples) {
            let append = model
                .encoder
                .encode_stream(&device, &mut stream_state, chunk, false)
                .unwrap();
            if let Some(hidden_states) = append.hidden_states {
                streamed_parts.push(hidden_states);
            }
        }
        let append = model
            .encoder
            .encode_stream(&device, &mut stream_state, &[], true)
            .unwrap();
        if let Some(hidden_states) = append.hidden_states {
            streamed_parts.push(hidden_states);
        }
        let streamed = Tensor::cat(streamed_parts, 1);
        let streamed = streamed.as_slice().await.unwrap();

        assert_eq!(full.shape(), streamed.shape());
        let max_diff = full
            .as_slice()
            .iter()
            .zip(streamed.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("streaming encoder max_diff vs full = {max_diff}");
        assert!(
            max_diff < 0.1,
            "streaming encoder diverged from full encoder on JFK: max_diff={max_diff}"
        );
    }

    #[tokio::test]
    async fn gpu_encoder_matches_cpu_on_zeros() {
        let Some((cpu_model, config, cpu_device)) = load_tiny_with_device(Device::cpu()) else {
            return;
        };
        let Ok(gpu_device) = Device::new().await else {
            return;
        };
        let Some((gpu_model, _, gpu_device)) = load_tiny_with_device(gpu_device) else {
            return;
        };

        let samples = vec![0.0f32; config.encoder_config.frame_len() * 80];
        let cpu = cpu_model.encoder.encode(&cpu_device, &samples).unwrap();
        let gpu = gpu_model.encoder.encode(&gpu_device, &samples).unwrap();

        let cpu = cpu.as_slice().await.unwrap();
        let gpu = gpu.as_slice().await.unwrap();
        assert_eq!(cpu.shape(), gpu.shape());
        let max_diff = cpu
            .as_slice()
            .iter()
            .zip(gpu.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("moonshine GPU encoder max_diff vs CPU = {max_diff}");
        assert!(
            max_diff < 0.1,
            "GPU encoder diverged from CPU encoder: max_diff={max_diff}"
        );
    }

    #[tokio::test]
    async fn gpu_cached_decode_matches_cpu_on_zeros() {
        let Some((mut cpu_model, config, cpu_device)) = load_tiny_with_device(Device::cpu()) else {
            return;
        };
        let Ok(gpu_device) = Device::new().await else {
            return;
        };
        let Some((mut gpu_model, _, gpu_device)) = load_tiny_with_device(gpu_device) else {
            return;
        };

        let samples = vec![0.0f32; config.encoder_config.frame_len() * 80];
        let cpu_encoder_hidden_states = cpu_model.encoder.encode(&cpu_device, &samples).unwrap();
        let gpu_encoder_hidden_states = gpu_model.encoder.encode(&gpu_device, &samples).unwrap();
        let cpu_prepared = cpu_model
            .decoder
            .prepare_encoder_hidden_states(&cpu_encoder_hidden_states)
            .unwrap();
        let gpu_prepared = gpu_model
            .decoder
            .prepare_encoder_hidden_states(&gpu_encoder_hidden_states)
            .unwrap();

        let tokens = [
            config.decoder_start_token(),
            123,
            456,
            config.eos_token_id.saturating_sub(1),
        ];
        let mut cpu_cache = MoonshineDecoderCache::default();
        let mut gpu_cache = MoonshineDecoderCache::default();
        let mut cpu_logits = cpu_model
            .decoder
            .decode_cached(&tokens[..1], &cpu_prepared, &mut cpu_cache)
            .unwrap();
        let mut gpu_logits = gpu_model
            .decoder
            .decode_cached(&tokens[..1], &gpu_prepared, &mut gpu_cache)
            .unwrap();
        for token in &tokens[1..] {
            cpu_logits = cpu_model
                .decoder
                .decode_cached(&[*token], &cpu_prepared, &mut cpu_cache)
                .unwrap();
            gpu_logits = gpu_model
                .decoder
                .decode_cached(&[*token], &gpu_prepared, &mut gpu_cache)
                .unwrap();
        }

        let cpu_last = cpu_logits.squeeze(0).squeeze(0).to_concrete();
        let gpu_last = gpu_logits.squeeze(0).squeeze(0).to_concrete();
        let cpu_last = cpu_last.as_slice().await.unwrap();
        let gpu_last = gpu_last.as_slice().await.unwrap();
        assert_eq!(cpu_last.shape(), gpu_last.shape());
        let max_diff = cpu_last
            .as_slice()
            .iter()
            .zip(gpu_last.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("moonshine GPU cached decode max_diff vs CPU = {max_diff}");
        assert!(
            max_diff < 0.1,
            "GPU cached decode diverged from CPU: max_diff={max_diff}"
        );
    }

    #[tokio::test]
    async fn cached_decode_matches_full_decode() {
        let Some((mut model, config, device)) = load_tiny_from_single_gguf() else {
            return;
        };

        let samples = vec![0.0f32; config.encoder_config.frame_len() * 80];
        let encoder_hidden_states = model.encoder.encode(&device, &samples).unwrap();
        let adapted_encoder_hidden_states = model
            .decoder
            .prepare_encoder_hidden_states(&encoder_hidden_states)
            .unwrap();

        let tokens = [
            config.decoder_start_token(),
            123,
            456,
            config.eos_token_id.saturating_sub(1),
        ];
        let full_logits = model
            .decoder
            .decode_prepared(&tokens, &adapted_encoder_hidden_states, None)
            .unwrap();
        let full_last = full_logits
            .narrow(1, tokens.len() - 1, 1)
            .squeeze(1)
            .squeeze(0)
            .to_concrete();
        let full_last = full_last.as_slice().await.unwrap();

        let mut cache = MoonshineDecoderCache::default();
        let mut cached_logits = model
            .decoder
            .decode_cached(&tokens[..1], &adapted_encoder_hidden_states, &mut cache)
            .unwrap();
        for token in &tokens[1..] {
            cached_logits = model
                .decoder
                .decode_cached(&[*token], &adapted_encoder_hidden_states, &mut cache)
                .unwrap();
        }
        let cached_last = cached_logits.squeeze(0).squeeze(0).to_concrete();
        let cached_last = cached_last.as_slice().await.unwrap();

        assert_eq!(full_last.shape(), cached_last.shape());
        let max_diff = full_last
            .as_slice()
            .iter()
            .zip(cached_last.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "cached decode diverged from full decode: max_diff={max_diff}"
        );
    }
}
