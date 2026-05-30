// Modified from https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/whisper/quantized_model.rs

use std::{num::NonZeroUsize, sync::Arc};

use fusor::{
    cache::{AttentionMask, KvCache, MaskCache, TensorCache},
    layers::{Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear},
    Device, Error, Result, Tensor, VarBuilder,
};
use timestamps::extract_timestamps;

use crate::config::Config;

pub(crate) mod timestamps;

fn conv1d(
    config: Conv1dConfig,
    device: &Device,
    vb: &mut VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
) -> Result<Conv1d<crate::WhisperDType>> {
    let weight_q = vb.get("weight", device)?;
    let weight_shape = weight_q.shape();

    // Handle both 2D and 3D weight formats
    let weight: Tensor<3, crate::WhisperDType> = if weight_shape.len() == 3 {
        // Already 3D: [out_channels, in_channels, kernel_size]
        weight_q.dequantize()
    } else {
        // 2D: reshape from [out_channels, in_channels*kernel_size] to [out_channels, in_channels, kernel_size]
        let weight_2d: Tensor<2, crate::WhisperDType> = weight_q.dequantize();
        weight_2d
            .reshape([out_channels, in_channels, kernel_size])
            .to_concrete()
    };

    let bias_q = vb.get("bias", device)?;
    let bias_shape = bias_q.shape();

    // Handle 1D, 2D bias formats
    let bias: Tensor<1, crate::WhisperDType> = if bias_shape.len() == 1 {
        bias_q.dequantize()
    } else {
        let bias_2d: Tensor<2, crate::WhisperDType> = bias_q.dequantize();
        // Squeeze to rank 1: assume shape is (1, out_channels) or (out_channels, 1)
        if bias_2d.shape()[0] == 1 {
            bias_2d.squeeze(0).to_concrete()
        } else {
            bias_2d.squeeze(1).to_concrete()
        }
    };
    Ok(Conv1d::new(weight, Some(bias), config))
}

struct MultiHeadAttentionCache {
    kv_cache: KvCache<crate::WhisperDType>,
}

impl MultiHeadAttentionCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            kv_cache: KvCache::new(1, max_seq_len),
        }
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
struct MultiHeadAttention {
    query: Linear<crate::WhisperDType>,
    key: Linear<crate::WhisperDType>,
    value: Linear<crate::WhisperDType>,
    out: Linear<crate::WhisperDType>,
    n_head: usize,
    softmax_span: tracing::Span,
    matmul_span: tracing::Span,
}

impl MultiHeadAttention {
    fn load(n_head: usize, device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let softmax_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-softmax");
        let matmul_span = tracing::span!(tracing::Level::TRACE, "multi-head-attn-matmul");
        let query = Linear::load(device, &mut vb.pp("q_proj"))?;
        let value = Linear::load(device, &mut vb.pp("v_proj"))?;
        let key = Linear::load(device, &mut vb.pp("k_proj"))?;
        let out = Linear::load(device, &mut vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
            softmax_span,
            matmul_span,
        })
    }

    fn forward_kv(
        &self,
        x: &Tensor<3, crate::WhisperDType>,
        cache: Option<&mut MultiHeadAttentionCache>,
    ) -> Result<(
        Tensor<3, crate::WhisperDType>,
        Tensor<3, crate::WhisperDType>,
    )> {
        let device = x.device();
        let key_states = self.key.forward(x);
        let value_states = self.value.forward(x);
        let (key_states, value_states) = match cache {
            None => (key_states, value_states),
            Some(cache) => {
                let key_states_4d = key_states.unsqueeze(2).to_concrete();
                let value_states_4d = value_states.unsqueeze(2).to_concrete();
                let (k, v) = cache
                    .kv_cache
                    .append(&device, &key_states_4d, &value_states_4d);
                (k.squeeze(2).to_concrete(), v.squeeze(2).to_concrete())
            }
        };
        Ok((key_states, value_states))
    }

    fn forward(
        &mut self,
        query: &Tensor<3, crate::WhisperDType>,
        kv: (
            Tensor<3, crate::WhisperDType>,
            Tensor<3, crate::WhisperDType>,
        ),
        mask: Option<&AttentionMask<crate::WhisperDType>>,
        attention_output: Option<&mut TensorCache<4, crate::WhisperDType>>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let query_states = self.query.forward(query);
        let (key_states, value_states) = &kv;
        let wv = self.qkv_attention(
            &query_states,
            key_states,
            value_states,
            mask,
            attention_output,
        )?;
        Ok(self.out.forward(&wv))
    }

    fn qkv_attention(
        &self,
        q: &Tensor<3, crate::WhisperDType>,
        k: &Tensor<3, crate::WhisperDType>,
        v: &Tensor<3, crate::WhisperDType>,
        mask: Option<&AttentionMask<crate::WhisperDType>>,
        attention_output: Option<&mut TensorCache<4, crate::WhisperDType>>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let device = q.device();
        let [n_batch, n_ctx_q, n_state] = q.shape();
        let [_, n_ctx_kv, _] = k.shape();
        let head_dim = n_state / self.n_head;
        let scale = crate::WhisperDType::from((head_dim as f32).powf(-0.25));

        // Reshape Q: [n_batch, n_ctx_q, n_state] -> [n_batch, n_head, n_ctx_q, head_dim]
        let q_target_dims = [n_batch, n_ctx_q, self.n_head, head_dim];
        let q_reshaped = q.reshape(q_target_dims);
        let q_transposed = q_reshaped.transpose(1, 2);
        let q = q_transposed.mul_scalar(scale);

        // Reshape K/V: [n_batch, n_ctx_kv, n_state] -> [n_batch, n_head, n_ctx_kv, head_dim]
        // For self-attention n_ctx_kv == n_ctx_q, for cross-attention they differ
        let kv_target_dims = [n_batch, n_ctx_kv, self.n_head, head_dim];
        let k_reshaped = k.reshape(kv_target_dims);
        let k_transposed = k_reshaped.transpose(1, 2);
        let k_transposed2 = k_transposed.transpose(2, 3);
        let k = k_transposed2.mul_scalar(scale);
        let v_reshaped = v.reshape(kv_target_dims);
        let v = v_reshaped.transpose(1, 2);

        let mut qk = {
            let _enter = self.matmul_span.enter();
            q.mat_mul(&k)
        };

        if let Some(mask) = mask {
            mask.forward(&mut qk)
        }
        if let Some(out) = attention_output {
            out.append(&device, &qk);
        }
        let w = {
            let _enter = self.softmax_span.enter();
            qk.softmax_last_dim_fused()
        };

        let wv_raw = {
            let _enter = self.matmul_span.enter();
            w.mat_mul(&v)
        };

        let wv = wv_raw.transpose(1, 2).flatten_last_n::<1, _>();

        Ok(wv)
    }
}

struct ResidualAttentionBlockCache {
    attn: MultiHeadAttentionCache,
    feature_attn_cache: Option<(
        Tensor<3, crate::WhisperDType>,
        Tensor<3, crate::WhisperDType>,
    )>,
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm<1, crate::WhisperDType>,
    cross_attn: Option<(MultiHeadAttention, LayerNorm<1, crate::WhisperDType>)>,
    mlp_linear1: Linear<crate::WhisperDType>,
    mlp_linear2: Linear<crate::WhisperDType>,
    mlp_ln: LayerNorm<1, crate::WhisperDType>,
    span: tracing::Span,
}

impl ResidualAttentionBlock {
    fn load(n_head: usize, cross_attn: bool, device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "residual-attn");
        let attn = MultiHeadAttention::load(n_head, device, &mut vb.pp("self_attn"))?;
        let attn_ln = LayerNorm::load(device, &mut vb.pp("self_attn_layer_norm"), 1e-5)?;
        let cross_attn = if cross_attn {
            let cross_attn = MultiHeadAttention::load(n_head, device, &mut vb.pp("encoder_attn"))?;
            let cross_attn_ln =
                LayerNorm::load(device, &mut vb.pp("encoder_attn_layer_norm"), 1e-5)?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let mlp_linear1 = Linear::load(device, &mut vb.pp("fc1"))?;
        let mlp_linear2 = Linear::load(device, &mut vb.pp("fc2"))?;
        let mlp_ln = LayerNorm::load(device, &mut vb.pp("final_layer_norm"), 1e-5)?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
            span,
        })
    }

    fn forward(
        &mut self,
        audio_features_kv: Option<(
            Tensor<3, crate::WhisperDType>,
            Tensor<3, crate::WhisperDType>,
        )>,
        x: &Tensor<3, crate::WhisperDType>,
        mask: Option<&AttentionMask<crate::WhisperDType>>,
        mut cache: Option<&mut ResidualAttentionBlockCache>,
        attention_output: Option<&mut TensorCache<4, crate::WhisperDType>>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let _enter = self.span.enter();

        let attn_ln_x = self.attn_ln.forward_fused(x);
        let kv = self
            .attn
            .forward_kv(&attn_ln_x, cache.as_mut().map(|cache| &mut cache.attn))?;
        let attn = self.attn.forward(&attn_ln_x, kv, mask, None)?;
        let mut x = (x + &attn).to_concrete();

        if let (Some(kv), Some((attn, ln))) = (audio_features_kv, &mut self.cross_attn) {
            let ln_x = ln.forward_fused(&x);
            let attn_out = attn.forward(&ln_x, kv, None, attention_output)?;
            x = (&x + &attn_out).to_concrete();
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward_fused(&x))
                .gelu(),
        );
        let result = (x + mlp).to_concrete();

        Ok(result)
    }
}

fn sinusoids(length: usize, channels: usize, device: &Device) -> Tensor<2, crate::WhisperDType> {
    let max_timescale = 10000f32;
    let log_timescale_increment = crate::WhisperDType::from(max_timescale.ln())
        / crate::WhisperDType::from((channels / 2 - 1) as f32);
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (crate::WhisperDType::from(i as f32) * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales_1d = Tensor::new(device, inv_timescales.as_slice());
    let inv_timescales = inv_timescales_1d.unsqueeze(0);
    let arange_u32 = fusor::arange(device, 0u32, length as u32);
    let arange_cast = arange_u32.cast::<crate::WhisperDType>();
    let arange = arange_cast.unsqueeze(1);
    let sh = [length, channels / 2];
    let arange_broadcast = arange.broadcast_as(sh);
    let inv_timescales_broadcast = inv_timescales.broadcast_as(sh);
    let scaled_time = (&arange_broadcast * &inv_timescales_broadcast).to_concrete();
    Tensor::cat(
        [
            scaled_time.sin().to_concrete(),
            scaled_time.cos().to_concrete(),
        ],
        1,
    )
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
pub struct AudioEncoder {
    conv1: Conv1d<crate::WhisperDType>,
    conv2: Conv1d<crate::WhisperDType>,
    positional_embedding: Tensor<2, crate::WhisperDType>,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm<1, crate::WhisperDType>,
    span: tracing::Span,
    conv1_span: tracing::Span,
    conv2_span: tracing::Span,
}

impl AudioEncoder {
    fn load(device: &Device, mut vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "audio-encoder");
        let conv1_span = tracing::span!(tracing::Level::TRACE, "conv1");
        let conv2_span = tracing::span!(tracing::Level::TRACE, "conv2");
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
            dilation: 1,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
            dilation: 1,
        };
        let n_mels = cfg.num_mel_bins;
        let conv1 = conv1d(cfg1, device, &mut vb.pp("conv1"), n_mels, n_state, 3)?;
        let conv2 = conv1d(cfg2, device, &mut vb.pp("conv2"), n_state, n_state, 3)?;
        let positional_embedding = sinusoids(n_ctx, n_state, device);
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(
                    n_head,
                    false,
                    device,
                    &mut vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = LayerNorm::load(device, &mut vb.pp("layer_norm"), 1e-5)?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
            conv1_span,
            conv2_span,
            span,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor<3, crate::WhisperDType>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let _enter = self.span.enter();

        let x = {
            let _enter = self.conv1_span.enter();
            self.conv1.forward(x).gelu()
        };
        let x = {
            let _enter = self.conv2_span.enter();
            self.conv2.forward(&x).gelu()
        };
        let x = x.transpose(1, 2);
        let [_bsize, seq_len, _hidden] = x.shape();

        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len);
        let mut x = x.add_(&positional_embedding);

        for block in self.blocks.iter_mut() {
            x = block.forward(None, &x, None, None, None)?;
        }
        let x = self.ln_post.forward_fused(&x);

        Ok(x)
    }
}

#[derive(Default)]
pub struct TextDecoderCache {
    tokens: Vec<u32>,
    blocks: Vec<ResidualAttentionBlockCache>,
}

impl TextDecoderCache {
    pub fn new() -> Self {
        Self::default()
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L176
pub struct TextDecoder {
    token_embedding: Embedding<crate::WhisperDType>,
    positional_embedding: Tensor<2, crate::WhisperDType>,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm<1, crate::WhisperDType>,
    max_target_positions: usize,
    mask_cache: Arc<MaskCache<crate::WhisperDType>>,
    span: tracing::Span,
    span_final: tracing::Span,
}

impl TextDecoder {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "text-decoder");
        let span_final = tracing::span!(tracing::Level::TRACE, "text-decoder-final");
        let n_head = cfg.decoder_attention_heads;
        let max_target_positions = cfg.max_target_positions;
        let token_embedding = Embedding::load(device, &mut vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get("embed_positions.weight", device)?.dequantize();
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(
                    n_head,
                    true,
                    device,
                    &mut vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = LayerNorm::load(device, &mut vb.pp("layer_norm"), 1e-5)?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            max_target_positions,
            mask_cache: Default::default(),
            span,
            span_final,
        })
    }

    pub fn forward(
        &mut self,
        tokens: &[u32],
        audio_features: &Tensor<2, crate::WhisperDType>,
        cache: &mut TextDecoderCache,
        mut attention_output: Option<&mut [TensorCache<4, crate::WhisperDType>]>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let index_pos = cache.tokens.len();
        cache.tokens.extend_from_slice(tokens);
        let seq_len = tokens.len();
        if index_pos + seq_len > self.max_target_positions {
            return Err(Error::msg("exceeded max sequence length"));
        }
        let device = audio_features.device();
        let mask = self.mask_cache.get_mask(seq_len, index_pos, None, &device);
        let x: Tensor<1, u32> = Tensor::new(&device, tokens);
        // The model expects a batch dim but this inference loop does not handle
        // it so we add it at this point.
        let x = x.unsqueeze(0).to_concrete();

        let _enter = self.span.enter();
        let token_embedding = self.token_embedding.forward(&x);
        let positional_embedding = self.positional_embedding.narrow(0, index_pos, seq_len);

        let mut x = token_embedding.add_(&positional_embedding);

        // Add batch dimension to audio_features for forward_kv
        let audio_features_batched = audio_features.unsqueeze(0).to_concrete();

        for (i, block) in self.blocks.iter_mut().enumerate() {
            if cache.blocks.len() <= i {
                cache.blocks.push(ResidualAttentionBlockCache {
                    attn: MultiHeadAttentionCache::new(self.max_target_positions),
                    feature_attn_cache: block
                        .cross_attn
                        .as_ref()
                        .and_then(|(attn, _)| attn.forward_kv(&audio_features_batched, None).ok()),
                });
            }
            let block_cache = &mut cache.blocks[i];
            let query = block_cache.feature_attn_cache.clone();
            let attention_output = attention_output.as_mut().map(|outputs| &mut outputs[i]);
            x = block.forward(query, &x, Some(&mask), Some(block_cache), attention_output)?;
        }

        let out = self.ln.forward_fused(&x);

        Ok(out)
    }

    pub fn final_linear(
        &self,
        x: &Tensor<3, crate::WhisperDType>,
    ) -> Result<Tensor<3, crate::WhisperDType>> {
        let embeddings = self.token_embedding.embeddings_quantized();

        let logits = {
            let _enter = self.span_final.enter();
            x.q_mat_mul(embeddings)
        };

        Ok(logits)
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L221
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(device: &Device, vb: &mut VarBuilder, config: Config) -> Result<Self> {
        let encoder = AudioEncoder::load(device, vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(device, &mut vb.pp("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub(crate) async fn dtw_timestamps(
        attention_heads: Option<&'static [[usize; 2]]>,
        filter_width: NonZeroUsize,
        n_frames: usize,
        mask: Vec<Vec<bool>>,
        attention_output: &[TensorCache<4, crate::WhisperDType>],
    ) -> Result<Vec<Vec<crate::WhisperDType>>> {
        let Some(attention_heads) = attention_heads else {
            panic!(
                "The attention heads for word-level timestamps are not available for this model",
            );
        };

        let mut attention_output_tensor = Vec::new();
        for attn in attention_output {
            attention_output_tensor.push(attn.current_data().unwrap().clone());
        }

        extract_timestamps(
            attention_heads,
            &attention_output_tensor,
            filter_width,
            n_frames,
            mask,
        )
        .await
    }
}
