use std::sync::Arc;

use crate::chat_template::HuggingFaceChatTemplate;
use crate::raw::attention_layer::LlamaAttention;
use crate::raw::rope::RopeImplementation;
use crate::LlamaSourceError;
use attention_layer::AttentionBias;
use attention_layer::AttentionVariant;
use attention_layer::FeedForwardVariant;
use attention_layer::GroupedAttention;
use attention_layer::LlamaFeedForward;
use attention_layer::PhiFeedForward;
use attention_layer::SeparateAttention;
use fusor::cache::MaskCache;
use fusor::layers::Embedding;
use fusor::layers::Linear;
use fusor::layers::RmsNorm;
use fusor::AddOp;
use fusor::CastTensor;
use fusor::CastTo;
use fusor::FloatDataType;
use fusor::FloatOps;
use fusor::MatmulImpl;
use fusor::MulOp;
use fusor::QMatrix;
use fusor::ShardedVarBuilder;
use fusor::SimdBinaryOp;
use fusor::SimdElement;
use fusor::SimdReduceOp;
use fusor::SumOp;
use fusor::{Device, Result, Tensor};
use fusor_gguf::GgufMetadata;
use fusor_gguf::GgufValue;

mod attention_layer;
pub mod cache;
mod rope;
mod vision;

use cache::LlamaCache;
use kalosm_language_model::MediaHints;

pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const GEMMA_DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

/// The configuration of a Llama model.
pub struct LlamaConfig<F: FloatDataType + SimdElement = f32> {
    pub(crate) rope_freq_weight: Option<Tensor<1, F>>,
    pub(crate) rope_theta: f32,
    pub(crate) context_length: usize,
    pub(crate) head_dimension: usize,
    n_head: usize,
    pub(crate) n_layer: usize,
    pub(crate) start_token_string: String,
    pub(crate) stop_token: u32,
    pub(crate) stop_token_string: String,
    pub(crate) chat_template: Option<HuggingFaceChatTemplate>,
    pub(crate) rope_scaling: Option<RopeScalingConfig>,
    pub(crate) sliding_window_type: Option<usize>,
    pub(crate) sliding_window_size: Option<usize>,
    pub(crate) vision_start_token: Option<u32>,
    pub(crate) _vision_end_token: Option<u32>,
    pub(crate) image_pad_token: Option<u32>,
    pub(crate) video_pad_token: Option<u32>,
    pub(crate) mrope_sections: Option<Vec<usize>>,
}

impl<F: FloatDataType + SimdElement> LlamaConfig<F> {
    fn hidden_size(&self) -> usize {
        self.head_dimension * self.n_head
    }

    #[cfg(test)]
    pub(crate) fn mock_test() -> Self {
        Self {
            rope_freq_weight: None,
            rope_theta: 5000.,
            context_length: 6,
            head_dimension: 2,
            n_head: 0,
            n_layer: 0,
            start_token_string: "<|startoftext|>".to_string(),
            stop_token: 0,
            stop_token_string: "<|endoftext|>".to_string(),
            sliding_window_type: None,
            sliding_window_size: None,
            chat_template: None,
            rope_scaling: None,
            vision_start_token: None,
            _vision_end_token: None,
            image_pad_token: None,
            video_pad_token: None,
            mrope_sections: None,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScalingConfig {
    pub(crate) factor: f32,
    pub(crate) high_freq_factor: f32,
    pub(crate) low_freq_factor: f32,
    pub(crate) original_max_position_embeddings: usize,
}

pub struct Model<F: FloatDataType + SimdElement = f32> {
    pub(crate) config: Arc<LlamaConfig<F>>,
    vision_encoder: Option<vision::QwenVisionTransformer<F>>,
    tok_embeddings: Embedding<F>,
    layers: Vec<LlamaAttention<F>>,
    norm: RmsNorm<1, F>,
    output: QMatrix,
    /// Mask cache always uses f32 for SIMD compatibility
    masks: MaskCache<f32>,
}

impl<F: FloatDataType + SimdElement + FloatOps + MatmulImpl> Model<F>
where
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        source: &mut ShardedVarBuilder<R>,
        vision_ct: Option<GgufMetadata>,
        vision_bytes: Option<Vec<u8>>,
        device: &Device,
        override_stop_token_string: Option<String>,
        override_chat_template: Option<String>,
        rope_scaling: Option<RopeScalingConfig>,
    ) -> std::result::Result<Self, LlamaSourceError>
    where
        f32: CastTensor<F> + CastTo<F>,
        F: CastTensor<f32> + CastTo<f32>,
    {
        // Helper to dequantize a QMatrix to 1D tensor
        // VarBuilder preserves original shapes, so 1D tensors stay 1D
        let dequantize_1d = |qmatrix: QMatrix| -> Tensor<1, F> {
            let shape = qmatrix.shape();
            if shape.len() == 1 {
                // Already 1D, dequantize directly
                let w1d: Tensor<1, f32> = qmatrix.dequantize();
                w1d.cast()
            } else if shape.len() == 2 {
                // 2D tensor, reshape to 1D (for backwards compatibility)
                let w2d: Tensor<2, f32> = qmatrix.dequantize();
                w2d.reshape([w2d.shape()[0] * w2d.shape()[1]])
                    .to_concrete()
                    .cast()
            } else {
                panic!(
                    "Expected 1D or 2D tensor for dequantize_1d, got {}D",
                    shape.len()
                )
            }
        };

        let decode_norm = |qmatrix: QMatrix, eps: f64| -> Result<RmsNorm<1, F>> {
            let weight = dequantize_1d(qmatrix);
            Ok(RmsNorm::new(weight, None, eps as f32))
        };

        // Get the eos and bos tokens from the metadata
        let tokens: Box<[GgufValue]> = source.get("tokenizer.ggml.tokens")?.clone().try_into()?;
        let tokens: Result<Vec<Box<str>>, LlamaSourceError> = tokens
            .iter()
            .map(|v| {
                let v: Box<str> = v.try_into()?;
                Ok(v)
            })
            .collect();
        let tokens = tokens?;
        let start_token: Option<u32> = source
            .get("tokenizer.ggml.bos_token_id")
            .ok()
            .and_then(|v| v.try_into().ok());
        let stop_token = if let Some(override_stop_token_string) = override_stop_token_string {
            tokens
                .iter()
                .position(|v| **v == override_stop_token_string)
                .unwrap_or(0) as u32
        } else {
            source
                .get("tokenizer.ggml.eos_token_id")?
                .clone()
                .try_into()?
        };
        let start_token_string = start_token
            .map(|v| tokens[v as usize].to_string())
            .unwrap_or_default();
        let stop_token_string = tokens[stop_token as usize].to_string();
        let chat_template = override_chat_template.or_else(|| {
            source
                .get("tokenizer.chat_template")
                .ok()
                .and_then(|v| v.to_string().ok())
                .map(|s| s.to_string())
        });
        let chat_template = match chat_template {
            Some(chat_template) => {
                let chat_template = HuggingFaceChatTemplate::create(chat_template)
                    .map_err(LlamaSourceError::ChatTemplate)?;
                Some(chat_template)
            }
            None => None,
        };

        // Parameter extraction from metadata.
        let architecture = source.get("general.architecture")?.to_string()?.clone();
        let head_count = source.get(".attention.head_count")?.to_u32()? as usize;
        let head_count_kv = source.get(".attention.head_count_kv")?.to_u32()? as usize;
        let block_count = source.get(".block_count")?.to_u32()? as usize;
        let embedding_length = source.get(".embedding_length")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = source.get(".attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = source
            .get(".rope.freq_base")
            .and_then(|m| Ok(m.to_f32()?))
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let sliding_window_size = source
            .get(".attention.sliding_window")
            .and_then(|m| Ok(m.to_u32()?))
            .ok()
            .map(|x| x as usize);
        let sliding_window_type = source
            .get(".attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()?))
            .ok()
            .map(|x| x as usize)
            .or_else(|| (&*architecture == "gemma3").then_some(GEMMA_DEFAULT_SLIDING_WINDOW_TYPE));

        let rope_freq_base_sliding = source
            .get(".rope.local_freq_base")
            .and_then(|m| Ok(m.to_f32()?))
            .ok()
            .or_else(|| {
                (&*architecture == "gemma3").then_some(GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING)
            });

        let context_length = source.get(".context_length")?.to_u32()? as usize;
        let head_dim = source
            .get(".attention.key_length")
            .and_then(|v| Ok(v.to_u32()?))
            .ok()
            .map(|x| x as usize)
            .unwrap_or_else(|| embedding_length / head_count);

        let rope_freq_weight: Option<Tensor<1, F>> = source
            .tensor("rope_freqs.weight", device)
            .ok()
            .map(&dequantize_1d);

        let config = LlamaConfig {
            rope_freq_weight,
            rope_theta: rope_freq_base,
            context_length,
            head_dimension: head_dim,
            n_head: head_count,
            n_layer: block_count,
            start_token_string,
            stop_token,
            stop_token_string,
            chat_template,
            rope_scaling,
            sliding_window_type,
            sliding_window_size,
            vision_start_token: tokens
                .iter()
                .position(|v| &**v == "<|vision_start|>")
                .map(|v| v as u32),
            _vision_end_token: tokens
                .iter()
                .position(|v| &**v == "<|vision_end|>")
                .map(|v| v as u32),
            image_pad_token: tokens
                .iter()
                .position(|v| &**v == "<|image_pad|>")
                .map(|v| v as u32),
            video_pad_token: tokens
                .iter()
                .position(|v| &**v == "<|video_pad|>")
                .map(|v| v as u32),
            mrope_sections: source
                .get(".rope.dimension_sections")
                .ok()
                .and_then(|m| {
                    m.to_array()
                        .ok()
                        .map(|v| v.iter().map(|x| x.to_i32().map(|x| x as usize)).collect())
                })
                .transpose()?,
        };
        let config = Arc::new(config);

        let rope: RopeImplementation<F> =
            rope::RopeImplementation::new(&config, config.rope_theta, device)?;
        let sliding_rope: Option<RopeImplementation<F>> = rope_freq_base_sliding
            .map(|rope_freq_base_sliding| {
                RopeImplementation::new(&config, rope_freq_base_sliding, device)
            })
            .transpose()?;

        let tok_embeddings_q = source.tensor("token_embd.weight", device)?;
        let mut tok_embeddings: Tensor<2, f32> = tok_embeddings_q.dequantize();
        // if this is gemma3, scale the tok_embeddings by sqrt(embedding_length)
        if &*architecture == "gemma3" {
            tok_embeddings = (tok_embeddings * (embedding_length as f32).sqrt()).to_concrete();
        }
        let tok_embeddings: Tensor<2, F> = tok_embeddings.cast();
        let tok_embeddings = Embedding::new_from_tensor(tok_embeddings);

        let norm = source.tensor("output_norm.weight", device)?;
        let norm = decode_norm(norm, rms_norm_eps)?;
        let output = source.tensor("output.weight", device).unwrap_or_else(|_| {
            // If there is no output layer, assume the word embeddings are tied to the output
            tok_embeddings_q.clone()
        });
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_variant = if let Ok(attention_qkv) =
                source.tensor(&format!("{prefix}.attn_qkv.weight"), device)
            {
                AttentionVariant::Grouped(GroupedAttention { attention_qkv })
            } else {
                let q = source.tensor(&format!("{prefix}.attn_q.weight"), device)?;
                let k = source.tensor(&format!("{prefix}.attn_k.weight"), device)?;
                let v = source.tensor(&format!("{prefix}.attn_v.weight"), device)?;
                let bias = if let (Ok(bias_q), Ok(bias_k), Ok(bias_v)) = (
                    source.tensor(&format!("{prefix}.attn_q.bias"), device),
                    source.tensor(&format!("{prefix}.attn_k.bias"), device),
                    source.tensor(&format!("{prefix}.attn_v.bias"), device),
                ) {
                    Some(AttentionBias::new(
                        dequantize_1d(bias_q),
                        dequantize_1d(bias_k),
                        dequantize_1d(bias_v),
                    ))
                } else {
                    None
                };
                let q_norm = source
                    .tensor(&format!("{prefix}.attn_q_norm.weight"), device)
                    .ok();
                let k_norm = source
                    .tensor(&format!("{prefix}.attn_k_norm.weight"), device)
                    .ok();
                let separate = SeparateAttention {
                    attention_wq: q,
                    attention_q_norm: q_norm
                        .map(|norm| decode_norm(norm, rms_norm_eps))
                        .transpose()?,
                    attention_wk: k,
                    attention_k_norm: k_norm
                        .map(|norm| decode_norm(norm, rms_norm_eps))
                        .transpose()?,
                    attention_wv: v,
                    interleaved_rope: architecture.as_ref() != "qwen2"
                        && architecture.as_ref() != "qwen3"
                        && architecture.as_ref() != "gemma3",
                    bias,
                };
                AttentionVariant::Separate(Box::new(separate))
            };
            let attention_wo = source.tensor(&format!("{prefix}.attn_output.weight"), device)?;
            // Try to read from the up, down and gate weights
            let feed_forward_variant = if let Ok(ffn_gate) =
                source.tensor(&format!("{prefix}.ffn_gate.weight"), device)
            {
                let feed_forward_w1 = ffn_gate;
                let feed_forward_w2 =
                    source.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 = source.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                FeedForwardVariant::Llama(Box::new(LlamaFeedForward::new(
                    feed_forward_w1,
                    feed_forward_w2,
                    feed_forward_w3,
                )))
            } else {
                // Otherwise, try to read from the up, and down weights
                let up = source.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                // Transpose the down tensor
                let down = source.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_length = source.get(".feed_forward_length")?.to_u32()? as usize;

                FeedForwardVariant::Phi(PhiFeedForward {
                    up,
                    down,
                    feed_forward_length,
                })
            };
            let attention_norm = source.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
            let post_attention_norm = source
                .tensor(&format!("{prefix}.post_attention_norm.weight"), device)
                .ok();
            let ffn_norm = source.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;
            let ffn_post_norm = source
                .tensor(&format!("{prefix}.post_ffw_norm.weight"), device)
                .ok();

            let mut layer_sliding_window_size = None;

            let rope_cache = if let (
                Some(rope_sliding),
                Some(sliding_window_type),
                Some(sliding_window_size),
            ) = (
                sliding_rope.as_ref(),
                sliding_window_type,
                sliding_window_size,
            ) {
                let is_sliding = (layer_idx + 1) % sliding_window_type != 0;
                if is_sliding {
                    layer_sliding_window_size = Some(sliding_window_size);
                    rope_sliding.clone()
                } else {
                    rope.clone()
                }
            } else {
                rope.clone()
            };

            layers.push(LlamaAttention {
                attention_variant,
                attention_wo: Linear::new(attention_wo, None),
                attention_norm: decode_norm(attention_norm, rms_norm_eps)?,
                post_attention_norm: post_attention_norm
                    .map(|norm| decode_norm(norm, rms_norm_eps))
                    .transpose()?,
                feed_forward_variant,
                ffn_norm: decode_norm(ffn_norm, rms_norm_eps)?,
                post_ffn_norm: ffn_post_norm
                    .map(|norm| decode_norm(norm, rms_norm_eps))
                    .transpose()?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                hidden_size: config.hidden_size(),
                rope_cache,
                sliding_window_size: layer_sliding_window_size,
            })
        }

        // If the model is a vision model, load the vision encoder
        let vision_encoder =
            if let (Some(vision_ct), Some(vision_bytes)) = (vision_ct, vision_bytes) {
                Some(vision::QwenVisionTransformer::from_gguf(
                    vision_ct,
                    &vision_bytes,
                    device,
                ))
            } else {
                None
            };
        Ok(Self {
            config,
            tok_embeddings,
            layers,
            norm,
            output,
            masks: Default::default(),
            vision_encoder: vision_encoder.transpose()?,
        })
    }
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> Model<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    #[allow(clippy::type_complexity)]
    pub fn encode_tokens(
        &self,
        raw_tokens: &[u32],
        raw_images: &[(image::DynamicImage, MediaHints)],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<(Tensor<3, F>, usize, usize, Option<Tensor<2, F>>)> {
        let mut grid_thw = Vec::new();
        let mut images = Vec::new();
        let mut image_token_ranges = Vec::new();
        // Embed all images
        if let Some(vision_encoder) = &self.vision_encoder {
            for (image, hints) in raw_images {
                let min_pixels = hints.min_tokens();
                let max_pixels = hints.max_tokens();
                let (image, thw) =
                    vision_encoder.preprocess_image(image, min_pixels, max_pixels)?;
                images.push(image);
                grid_thw.push(thw)
            }
        }

        // Add any image padding tokens to the tokens if needed
        let tokens = if let (Some(image_pad_token), Some(vision_start_token), Some(vision)) = (
            self.config.image_pad_token,
            self.config.vision_start_token,
            &self.vision_encoder,
        ) {
            let mut tokens = Vec::new();
            let mut token_iter = raw_tokens.iter().copied();
            let mut image_iter = grid_thw.iter();
            while let Some(token) = token_iter.next() {
                tokens.push(token);
                let start_index = tokens.len();
                if token == vision_start_token {
                    match token_iter.next() {
                        Some(next) if next == image_pad_token => {
                            // Push a pad token for every image token
                            let grid = image_iter.next().ok_or_else(|| {
                                fusor::Error::msg("Image pad token found without matching image.")
                            })?;
                            for _ in 0..grid.iter().product::<u32>()
                                / (vision.spacial_merge_size as u32).pow(2)
                            {
                                tokens.push(image_pad_token);
                            }
                            image_token_ranges.push(start_index..tokens.len());
                        }
                        Some(next) => {
                            tokens.push(next);
                        }
                        None => break,
                    }
                }
            }
            tokens
        } else {
            raw_tokens.to_vec()
        };

        let mut seq_len = tokens.len();
        let cached_tokens = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
        // We use a lower cutoff than the context length to avoid recomputing the attention every single token
        let cutoff_len: usize = self.config.context_length.saturating_sub(32).max(8);
        let (tokens, index_pos, start_time) = if seq_len + cached_tokens
            > self.config.context_length
        {
            let all_tokens = if let Some(cache) = cache.as_mut() {
                cache.clear();
                let mut all_tokens = cache.tokens.clone();
                all_tokens.extend(tokens);
                all_tokens
            } else {
                tokens.to_vec()
            };
            let start = all_tokens.len() - cutoff_len;
            seq_len = cutoff_len;
            tracing::trace!("The context is full, trimming start of the context to fit new tokens. The first {} tokens were truncated.", start);
            let all_tokens = &all_tokens[start..];
            if let Some(cache) = cache.as_mut() {
                cache.tokens = all_tokens.to_vec();
            }
            assert!(all_tokens.len() <= self.config.context_length);
            (all_tokens.to_vec(), 0, 0)
        } else {
            let index_pos = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
            let start_time = cache.as_ref().map(|c| c.start_time).unwrap_or_default();
            if let Some(cache) = cache.as_mut() {
                cache.tokens.extend_from_slice(&tokens);
            }
            (tokens, index_pos, start_time)
        };
        let x_base = Tensor::new(device, tokens.as_slice());
        let x = x_base.unsqueeze(0);

        let mut embeddings = self.tok_embeddings.forward(&x);
        let mut pos_ids = None;
        let batch_size = embeddings.shape()[0];
        let embed_dim = embeddings.shape()[2];

        if let Some(vision_encoder) = &self.vision_encoder {
            for ((pixels, grid), range) in images.iter().zip(&grid_thw).zip(image_token_ranges) {
                let pixels_f: Tensor<2, F> = pixels.cast();
                let image_embeds = vision_encoder.forward_image(&pixels_f, *grid)?;
                embeddings = embeddings.slice_assign(
                    [0..batch_size, range, 0..embed_dim],
                    &image_embeds.unsqueeze(0),
                );
            }
            let (new_pos_ids, new_start_time) =
                vision_encoder.get_rope_index(&tokens, &grid_thw, &self.config, start_time)?;
            if let Some(cache) = cache.as_mut() {
                cache.start_time = new_start_time;
            }
            let pos_f32: Tensor<2, f32> = new_pos_ids.cast();
            let pos_f: Tensor<2, F> = pos_f32.cast();
            pos_ids = Some(pos_f);
        }

        Ok((embeddings, seq_len, index_pos, pos_ids))
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        images: &[(image::DynamicImage, MediaHints)],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2, F>>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        let (mut layer_in, seq_len, index_pos, pos_ids) =
            self.encode_tokens(tokens, images, device, cache.as_deref_mut())?;

        for (i, layer) in self.layers.iter().enumerate() {
            let x = layer_in;
            let residual: Tensor<3, f32> = x.cast();
            let x = layer.attention_norm.forward_generic(&x);
            let mask = self
                .masks
                .get_mask(seq_len, index_pos, layer.sliding_window_size, device);
            let mut attn = layer.forward(
                &x,
                Some(&mask),
                index_pos,
                pos_ids.as_ref(),
                cache.as_mut().map(|c| &mut c.blocks[i]),
            );
            if let Some(post_attention_norm) = &layer.post_attention_norm {
                attn = post_attention_norm.forward_generic(&attn);
            }
            // Work in f32 for tensor addition
            let attn_f32: Tensor<3, f32> = attn.cast();
            let x_f32: Tensor<3, F> = (attn_f32 + residual).cast();

            // MLP
            let residual: Tensor<3, f32> = x_f32.cast();
            let x = layer.ffn_norm.forward_generic(&x_f32);
            let mut x = layer.feed_forward_variant.forward(&x);
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                x = post_ffn_norm.forward_generic(&x);
            }
            // Work in f32 for tensor addition
            let x_f32: Tensor<3, f32> = x.cast();
            layer_in = (x_f32 + residual).cast();
        }
        let x = self.norm.forward_generic(&layer_in);
        let x = x.i((.., seq_len - 1, ..));
        // Do final matmul in f32 and cast back
        let x_f32 = x.cast::<f32>();
        let result_f32 = x_f32.q_mat_mul(&self.output);
        Ok(result_f32.cast())
    }
}
