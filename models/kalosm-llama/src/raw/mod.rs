use std::sync::{Arc, Mutex};

use crate::chat_template::HuggingFaceChatTemplate;
use crate::raw::attention_layer::LlamaAttention;
use crate::raw::rope::{RopeAt, RopeImplementation};
use crate::{LlamaImage, LlamaSourceError};
use attention_layer::AttentionBias;
use attention_layer::AttentionVariant;
use attention_layer::FeedForwardVariant;
use attention_layer::GroupedAttention;
use attention_layer::LlamaFeedForward;
use attention_layer::PhiFeedForward;
use attention_layer::SeparateAttention;
use fusor::cache::{MaskCache, MaskKind};
use fusor::layers::RmsNorm;
use fusor::{Device, Dim, Dtype, Graph, Result, Tensor};
use fusor_gguf::{GgufValue, RawTensorBytes, ShardedVarBuilder};
use weight::Weight;

mod attention_layer;
pub mod cache;
mod rope;
#[cfg(feature = "vision")]
mod vision;
mod weight;

/// One token's rope position on the `(time, height, width)` axes; every
/// axis agrees for a text token.
#[cfg(feature = "vision")]
pub(crate) use vision::RopePosition;
#[cfg(not(feature = "vision"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RopePosition(pub(crate) [u32; 3]);
#[cfg(not(feature = "vision"))]
impl RopePosition {
    pub(crate) fn text(p: u32) -> Self {
        Self([p; 3])
    }
    pub(crate) fn scalar(self) -> Option<u32> {
        let [t, h, w] = self.0;
        (t == h && h == w).then_some(t)
    }
}

use cache::LlamaCache;

pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const GEMMA_DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

/// The configuration of a Llama model.
pub struct LlamaConfig {
    pub(crate) rope_freq_weight: Option<Vec<f32>>,
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
    #[allow(dead_code)]
    pub(crate) sliding_window_type: Option<usize>,
    #[allow(dead_code)]
    pub(crate) sliding_window_size: Option<usize>,
    /// The multi-axis rope's frequency sections (Qwen-VL), or `None`.
    pub(crate) mrope_sections: Option<Vec<usize>>,
    pub(crate) vision_start_token: Option<u32>,
    pub(crate) image_pad_token: Option<u32>,
}

impl LlamaConfig {
    fn hidden_size(&self) -> usize {
        self.head_dimension * self.n_head
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScalingConfig {
    pub(crate) factor: f32,
    pub(crate) high_freq_factor: f32,
    pub(crate) low_freq_factor: f32,
    pub(crate) original_max_position_embeddings: usize,
}

pub struct Model {
    pub(crate) config: Arc<LlamaConfig>,
    tok_embeddings: Weight,
    tok_embedding_scale: Option<f32>,
    layers: Vec<LlamaAttention>,
    norm: RmsNorm,
    output: Weight,
    /// Memoizes the materialized (rectangular / windowed) masks.
    masks: Mutex<MaskCache>,
    /// The decode loop's persistent input leaves: the token id and its
    /// absolute position, both `[1]` `u32`. Only their *bytes* change per
    /// step, so every step reuses one graph and replays one plan.
    step_inputs: std::sync::OnceLock<(Tensor<1, u32>, Tensor<1, u32>)>,
    /// The embedding-row step's persistent leaves: one `[1, 1, hidden]`
    /// embedding and its `[1, head_dim / 2]` cos and sin rows. Only their
    /// bytes change per step, so an image prompt's tokens all replay one
    /// graph.
    embed_inputs: std::sync::OnceLock<(Tensor<3>, Tensor<2>, Tensor<2>)>,
    #[cfg(feature = "vision")]
    vision_encoder: Option<vision::QwenVisionTransformer>,
}

/// Each image's token range in the expanded prompt and its embeddings.
type ImageEmbeds = Vec<(std::ops::Range<usize>, Tensor<2>)>;

/// The embedded token inputs produced by [`Model::encode_tokens`], ready to be
/// run through the transformer layers.
pub(crate) struct EncodedTokens {
    embeddings: Tensor<3>,
    seq_len: usize,
    index_pos: usize,
    /// One rope position per token. `None` when every token sits at
    /// `index_pos + i`, which is every text-only model.
    positions: Option<Vec<RopePosition>>,
}

/// The `(cos, sin)` tables of an encoded sequence's positions, when any of
/// them needs more than a table row.
fn tables_for(
    rope: &RopeImplementation,
    positions: Option<&[RopePosition]>,
    device: &Device,
) -> Option<(Tensor<2>, Tensor<2>)> {
    let positions = positions?;
    if positions.iter().all(|p| p.scalar().is_some()) {
        return None;
    }
    Some(rope.tables_for(positions, device))
}

pub(crate) trait LlamaVarSource {
    fn get(&self, name: &str) -> Result<&GgufValue>;
    fn tensor(&self, name: &str) -> Result<RawTensorBytes>;
}

impl LlamaVarSource for ShardedVarBuilder {
    fn get(&self, name: &str) -> Result<&GgufValue> {
        ShardedVarBuilder::get(self, name)
    }

    fn tensor(&self, name: &str) -> Result<RawTensorBytes> {
        ShardedVarBuilder::tensor(self, name)
    }
}

/// A GGUF tensor as a dense rank-1 `f32` value (norm weights, biases,
/// `rope_freqs.weight`).
pub(crate) fn dense_1d(device: &Device, raw: &RawTensorBytes) -> Result<Tensor<1>> {
    let n: u64 = raw.shape.iter().product();
    match raw.fmt {
        // The dtype is data read out of the file; the rank is not.
        Dtype::F32 | Dtype::F16 => Ok(Tensor::from_raw_bytes(
            device,
            raw.fmt,
            [Dim::Const(n)],
            &raw.bytes,
        )),
        Dtype::Q(_) => Ok(Weight::from_raw(device.graph(), raw)?
            .quantized()
            .expect("a Q dtype loads as a quantized weight")
            .to_tensor()
            .reshape_dims([Dim::Const(n)])),
        other => Err(fusor::Error::Dtype(format!(
            "{} has dtype {other:?}, which has no dense 1d path",
            raw.name
        ))),
    }
}

impl Model {
    pub fn from_gguf<S: LlamaVarSource>(
        source: &mut S,
        vision_bytes: Option<Vec<u8>>,
        device: &Device,
        override_stop_token_string: Option<String>,
        override_chat_template: Option<String>,
        rope_scaling: Option<RopeScalingConfig>,
    ) -> std::result::Result<Self, LlamaSourceError> {
        let graph = device.graph().clone();

        let decode_norm = |raw: RawTensorBytes, eps: f64| -> Result<RmsNorm> {
            Ok(RmsNorm::new(Some(dense_1d(device, &raw)?), eps as f32))
        };

        // Get the eos and bos tokens from the metadata
        let tokens: Vec<String> = source
            .get("tokenizer.ggml.tokens")?
            .to_array()?
            .iter()
            .map(|v| Ok(v.to_string_value()?.to_string()))
            .collect::<Result<_>>()?;
        let start_token: Option<u32> = source
            .get("tokenizer.ggml.bos_token_id")
            .ok()
            .and_then(|v| v.to_u32().ok());
        let stop_token = if let Some(override_stop_token_string) = override_stop_token_string {
            tokens
                .iter()
                .position(|v| **v == override_stop_token_string)
                .unwrap_or(0) as u32
        } else {
            source.get("tokenizer.ggml.eos_token_id")?.to_u32()?
        };
        let start_token_string = start_token
            .map(|v| tokens[v as usize].to_string())
            .unwrap_or_default();
        let stop_token_string = tokens[stop_token as usize].to_string();
        let chat_template = override_chat_template.or_else(|| {
            source
                .get("tokenizer.chat_template")
                .ok()
                .and_then(|v| v.as_str())
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
        let architecture = source
            .get("general.architecture")?
            .to_string_value()?
            .to_string();
        let head_count = source.get(".attention.head_count")?.to_u32()? as usize;
        let head_count_kv = source.get(".attention.head_count_kv")?.to_u32()? as usize;
        let block_count = source.get(".block_count")?.to_u32()? as usize;
        let embedding_length = source.get(".embedding_length")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = source.get(".attention.layer_norm_rms_epsilon")?.to_f32()? as f64;

        let rope_freq_base = source
            .get(".rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let sliding_window_size = source
            .get(".attention.sliding_window")
            .and_then(|m| m.to_u32())
            .ok()
            .map(|x| x as usize);
        let sliding_window_type = source
            .get(".attention.sliding_window_type")
            .and_then(|m| m.to_u32())
            .ok()
            .map(|x| x as usize)
            .or_else(|| (&*architecture == "gemma3").then_some(GEMMA_DEFAULT_SLIDING_WINDOW_TYPE));

        let rope_freq_base_sliding = source
            .get(".rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .ok()
            .or_else(|| {
                (&*architecture == "gemma3").then_some(GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING)
            });

        // A multi-axis rope (Qwen-VL): frequency sections per position axis.
        let mrope_sections: Option<Vec<usize>> = source
            .get(".rope.dimension_sections")
            .ok()
            .and_then(|v| v.to_array().ok())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.to_u32().ok())
                    .filter(|&n| n > 0)
                    .map(|n| n as usize)
                    .collect()
            });
        let token_id = |text: &str| tokens.iter().position(|v| v == text).map(|i| i as u32);
        let vision_start_token = token_id("<|vision_start|>");
        let image_pad_token = token_id("<|image_pad|>");

        let context_length = source.get(".context_length")?.to_u32()? as usize;
        let head_dim = source
            .get(".attention.key_length")
            .and_then(|v| v.to_u32())
            .ok()
            .map(|x| x as usize)
            .unwrap_or_else(|| embedding_length / head_count);

        let rope_freq_weight: Option<Vec<f32>> = match source.tensor("rope_freqs.weight") {
            Ok(raw) => Some(dense_1d(device, &raw)?.to_vec_f32()),
            Err(_) => None,
        };

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
            mrope_sections,
            vision_start_token,
            image_pad_token,
        };
        let config = Arc::new(config);

        let rope = RopeImplementation::new(&config, config.rope_theta, device);
        let sliding_rope = rope_freq_base_sliding.map(|rope_freq_base_sliding| {
            RopeImplementation::new(&config, rope_freq_base_sliding, device)
        });

        let tok_embeddings_q = Weight::from_raw(&graph, &source.tensor("token_embd.weight")?)?;
        let tok_embedding_scale =
            (&*architecture == "gemma3").then(|| (embedding_length as f32).sqrt());

        let norm = source.tensor("output_norm.weight")?;
        let norm = decode_norm(norm, rms_norm_eps)?;
        let output = match source.tensor("output.weight") {
            Ok(output) => Weight::from_raw(&graph, &output)?,
            // If there is no output layer, assume the word embeddings are tied to the output
            Err(_) => tok_embeddings_q.clone(),
        };
        let mut layers = Vec::with_capacity(block_count);
        let interleaved_rope = !matches!(
            architecture.as_str(),
            "qwen2" | "qwen2vl" | "qwen3" | "gemma3"
        );
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_variant = if let Ok(attention_qkv) =
                source.tensor(&format!("{prefix}.attn_qkv.weight"))
            {
                AttentionVariant::Grouped(GroupedAttention {
                    attention_qkv: Weight::from_raw(&graph, &attention_qkv)?,
                    interleaved_rope,
                })
            } else {
                let q =
                    Weight::from_raw(&graph, &source.tensor(&format!("{prefix}.attn_q.weight"))?)?;
                let k =
                    Weight::from_raw(&graph, &source.tensor(&format!("{prefix}.attn_k.weight"))?)?;
                let v =
                    Weight::from_raw(&graph, &source.tensor(&format!("{prefix}.attn_v.weight"))?)?;
                let qkv = Weight::concat_rows(&[&q, &k, &v]);
                let bias_q = source.tensor(&format!("{prefix}.attn_q.bias"));
                let bias_k = source.tensor(&format!("{prefix}.attn_k.bias"));
                let bias_v = source.tensor(&format!("{prefix}.attn_v.bias"));
                let bias = if let (Ok(bias_q), Ok(bias_k), Ok(bias_v)) = (bias_q, bias_k, bias_v) {
                    Some(AttentionBias::new(
                        dense_1d(device, &bias_q)?,
                        dense_1d(device, &bias_k)?,
                        dense_1d(device, &bias_v)?,
                    ))
                } else {
                    None
                };
                let q_norm = source.tensor(&format!("{prefix}.attn_q_norm.weight")).ok();
                let k_norm = source.tensor(&format!("{prefix}.attn_k_norm.weight")).ok();
                let separate = SeparateAttention {
                    attention_wq: q,
                    attention_qkv: qkv,
                    attention_q_norm: q_norm
                        .map(|norm| decode_norm(norm, rms_norm_eps))
                        .transpose()?,
                    attention_wk: k,
                    attention_k_norm: k_norm
                        .map(|norm| decode_norm(norm, rms_norm_eps))
                        .transpose()?,
                    attention_wv: v,
                    interleaved_rope,
                    bias,
                };
                AttentionVariant::Separate(Box::new(separate))
            };
            let attention_wo = Weight::from_raw(
                &graph,
                &source.tensor(&format!("{prefix}.attn_output.weight"))?,
            )?;
            // Try to read from the up, down and gate weights
            let feed_forward_variant = if let Ok(ffn_gate) =
                source.tensor(&format!("{prefix}.ffn_gate.weight"))
            {
                let feed_forward_w1 = Weight::from_raw(&graph, &ffn_gate)?;
                let feed_forward_w2 = Weight::from_raw(
                    &graph,
                    &source.tensor(&format!("{prefix}.ffn_down.weight"))?,
                )?;
                let feed_forward_w3 =
                    Weight::from_raw(&graph, &source.tensor(&format!("{prefix}.ffn_up.weight"))?)?;
                FeedForwardVariant::Llama(Box::new(LlamaFeedForward::new(
                    feed_forward_w1,
                    feed_forward_w2,
                    feed_forward_w3,
                )))
            } else {
                // Otherwise, try to read from the up, and down weights
                let up =
                    Weight::from_raw(&graph, &source.tensor(&format!("{prefix}.ffn_up.weight"))?)?;
                let down = Weight::from_raw(
                    &graph,
                    &source.tensor(&format!("{prefix}.ffn_down.weight"))?,
                )?;
                let feed_forward_length = source.get(".feed_forward_length")?.to_u32()? as usize;

                FeedForwardVariant::Phi(PhiFeedForward {
                    up,
                    down,
                    feed_forward_length,
                })
            };
            let attention_norm = source.tensor(&format!("{prefix}.attn_norm.weight"))?;
            let post_attention_norm = source
                .tensor(&format!("{prefix}.post_attention_norm.weight"))
                .ok();
            let ffn_norm = source.tensor(&format!("{prefix}.ffn_norm.weight"))?;
            let ffn_post_norm = source
                .tensor(&format!("{prefix}.post_ffw_norm.weight"))
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
                attention_wo,
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

        #[cfg(feature = "vision")]
        let vision_encoder = vision_bytes
            .map(|bytes| vision::QwenVisionTransformer::from_gguf(bytes, device))
            .transpose()?;
        #[cfg(not(feature = "vision"))]
        let _ = vision_bytes;

        Ok(Self {
            config,
            tok_embeddings: tok_embeddings_q,
            tok_embedding_scale,
            layers,
            norm,
            output,
            masks: Mutex::new(MaskCache::new()),
            step_inputs: std::sync::OnceLock::new(),
            embed_inputs: std::sync::OnceLock::new(),
            #[cfg(feature = "vision")]
            vision_encoder,
        })
    }

    /// The context-window bookkeeping half of `encode_tokens`: which tokens
    /// to run and at which absolute starting position, with the cache's
    /// token record updated (and the cache cleared when the window overflows).
    fn plan_tokens(
        &self,
        raw_tokens: &[u32],
        mut cache: Option<&mut LlamaCache>,
    ) -> (Vec<u32>, usize) {
        let tokens = raw_tokens.to_vec();
        let mut seq_len = tokens.len();
        let cached_tokens = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
        // We use a lower cutoff than the context length to avoid recomputing the attention every single token
        let cutoff_len: usize = self.config.context_length.saturating_sub(32).max(8);
        let (tokens, index_pos) = if seq_len + cached_tokens > self.config.context_length {
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
            tracing::trace!(
                "The context is full, trimming start of the context to fit new tokens. The first {} tokens were truncated.",
                start
            );
            let all_tokens = &all_tokens[start..];
            if let Some(cache) = cache.as_mut() {
                cache.tokens = all_tokens.to_vec();
                cache.rope_position = 0;
            }
            assert!(all_tokens.len() <= self.config.context_length);
            (all_tokens.to_vec(), 0)
        } else {
            let index_pos = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
            if let Some(cache) = cache.as_mut() {
                cache.tokens.extend_from_slice(&tokens);
            }
            (tokens, index_pos)
        };
        let _ = seq_len;
        (tokens, index_pos)
    }

    /// Expand every `<|vision_start|><|image_pad|>` pair into one pad token
    /// per merged image token, and embed the images. Returns the expanded
    /// tokens, each image's `(token range, embeddings)` and its merged grid.
    #[cfg(feature = "vision")]
    #[allow(clippy::type_complexity)]
    fn expand_images(
        &self,
        raw_tokens: &[u32],
        images: &[LlamaImage],
    ) -> Result<(Vec<u32>, ImageEmbeds, Vec<[u32; 2]>)> {
        let (Some(vision), Some(start_token), Some(pad_token)) = (
            &self.vision_encoder,
            self.config.vision_start_token,
            self.config.image_pad_token,
        ) else {
            return Ok((raw_tokens.to_vec(), Vec::new(), Vec::new()));
        };
        let mut encoded = Vec::with_capacity(images.len());
        for (image, hints) in images {
            let (patches, grid) =
                vision.preprocess_image(image, hints.min_tokens(), hints.max_tokens());
            encoded.push((vision.forward_image(&patches, grid)?, grid));
        }
        let mut encoded = encoded.into_iter();
        let mut tokens = Vec::with_capacity(raw_tokens.len());
        let mut ranges = Vec::new();
        let mut grids = Vec::new();
        let mut it = raw_tokens.iter().copied().peekable();
        while let Some(token) = it.next() {
            tokens.push(token);
            if token == start_token && it.peek() == Some(&pad_token) {
                let Some((embeds, grid)) = encoded.next() else {
                    return Err(fusor::Error::Shape(
                        "an image placeholder in the prompt has no image".into(),
                    ));
                };
                it.next();
                let start = tokens.len();
                let n = vision.tokens_for(grid);
                tokens.extend(std::iter::repeat_n(pad_token, n));
                ranges.push((start..start + n, embeds));
                let m = vision.spatial_merge_size as u32;
                grids.push([grid[1] / m, grid[2] / m]);
            }
        }
        Ok((tokens, ranges, grids))
    }

    pub fn encode_tokens(
        &self,
        raw_tokens: &[u32],
        images: &[LlamaImage],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<EncodedTokens> {
        #[cfg(feature = "vision")]
        let (expanded, image_embeds, grids) = self.expand_images(raw_tokens, images)?;
        #[cfg(not(feature = "vision"))]
        let (expanded, image_embeds, grids): (Vec<u32>, ImageEmbeds, Vec<[u32; 2]>) = {
            let _ = images;
            (raw_tokens.to_vec(), Vec::new(), Vec::new())
        };
        let (tokens, index_pos) = self.plan_tokens(&expanded, cache.as_deref_mut());
        let seq_len = tokens.len();
        // A trimmed window no longer lines up with the image ranges; the
        // pads then embed as ordinary tokens.
        let image_embeds = if seq_len == expanded.len() {
            image_embeds
        } else {
            Vec::new()
        };
        let ids = Tensor::from_slice(device, [seq_len], &tokens);
        let mut text = self.tok_embeddings.rows_at(&ids);
        if let Some(scale) = self.tok_embedding_scale {
            text = text.mul_scalar(scale);
        }
        let embeddings = if image_embeds.is_empty() {
            text.unsqueeze(0)
        } else {
            let mut pieces = Vec::with_capacity(2 * image_embeds.len() + 1);
            let mut at = 0;
            for (range, embeds) in image_embeds {
                if range.start > at {
                    pieces.push(text.narrow(0, at, range.start - at));
                }
                pieces.push(embeds);
                at = range.end;
            }
            if at < seq_len {
                pieces.push(text.narrow(0, at, seq_len - at));
            }
            // One vision pass: materialized here rather than re-derived by
            // every step that narrows a row out of it.
            Tensor::cat(pieces, 0).materialize().unsqueeze(0)
        };

        let start = cache
            .as_deref()
            .map_or(index_pos as u32, |c| c.rope_position);
        let positions = match (
            self.config.mrope_sections.as_ref(),
            self.config.vision_start_token,
            self.config.image_pad_token,
        ) {
            (Some(_), Some(start_token), Some(pad_token)) => {
                #[cfg(feature = "vision")]
                let (positions, next) =
                    vision::rope_index(&tokens, start_token, pad_token, &grids, start);
                #[cfg(not(feature = "vision"))]
                let (positions, next) = {
                    let _ = (start_token, pad_token, grids);
                    let positions: Vec<RopePosition> = (0..seq_len as u32)
                        .map(|i| RopePosition::text(start + i))
                        .collect();
                    (positions, start + seq_len as u32)
                };
                if let Some(cache) = cache {
                    cache.rope_position = next;
                }
                Some(positions)
            }
            _ => {
                if let Some(cache) = cache {
                    cache.rope_position = start + seq_len as u32;
                }
                None
            }
        };

        Ok(EncodedTokens {
            embeddings,
            seq_len,
            index_pos,
            positions,
        })
    }

    /// The `(MaskKind, mask tensor)` a `[q_len, k_len]` score block needs.
    fn mask_for(
        &self,
        graph: &Graph,
        q_len: usize,
        k_len: usize,
        window: Option<usize>,
    ) -> Result<(MaskKind, Option<Tensor<2>>)> {
        if q_len == 1 {
            // One query against a warm cache sees every remaining key (a
            // sliding window is enforced by eviction).
            return Ok((MaskKind::None, None));
        }
        if q_len == k_len && window.is_none_or(|w| w >= k_len) {
            return Ok((MaskKind::Causal, None));
        }
        let mask = self.masks.lock().unwrap().materialized(
            graph,
            Dim::Const(q_len as u64),
            Dim::Const(k_len as u64),
            window.map(|w| w as u64),
        )?;
        Ok((MaskKind::QkMask, mask.tensor().cloned()))
    }

    /// The last token's logits as `[1, vocab]`. Deliberately NOT reshaped to
    /// rank 1: the reshape is a pure `Restride` view, and a view cannot be
    /// the root of a resolve on its own (nothing would land in a buffer).
    /// The row-major bytes are identical either way.
    /// The fixed-cache token loop: every id is one replayed decode step.
    fn forward_fixed_tokens(
        &self,
        tokens: &[u32],
        device: &Device,
        cache: &mut LlamaCache,
    ) -> Result<Tensor<2>> {
        let (steps, _index_pos) = self.plan_tokens(tokens, Some(cache));
        let n = steps.len();
        let mut logits = None;
        let base = cache.rope_position as usize;
        cache.rope_position += n as u32;
        for (i, tok) in steps.iter().enumerate() {
            let want_logits = i + 1 == n;
            let out = self.decode_step(*tok, base + i, device, cache, want_logits);
            self.commit_step(out.as_ref(), device, cache)?;
            logits = out.or(logits);
        }
        logits.ok_or_else(|| fusor::Error::Shape("forward of no tokens".into()))
    }

    /// The fixed-cache embedding loop an image prompt takes.
    fn forward_fixed_embeds(
        &self,
        tokens: &[u32],
        images: &[LlamaImage],
        device: &Device,
        cache: &mut LlamaCache,
    ) -> Result<Tensor<2>> {
        // Image tokens are embeddings, not ids, and sit at multi-axis
        // rope positions. Every token of the prompt — text included,
        // so the caches' armed appends stay with one graph — runs as
        // an embedding-row step: the row and its rope rows are leaf
        // bytes, so the whole prompt replays one graph. The token
        // graph's memo is stale after this and rebuilds on the next
        // text token.
        cache.decode_graph = None;
        let encoded = self.encode_tokens(tokens, images, device, Some(cache))?;
        let rope = &self.layers[0].rope_cache;
        let n = encoded.seq_len;
        let positions: Vec<RopePosition> = match encoded.positions {
            Some(p) => p,
            None => (0..n)
                .map(|i| RopePosition::text((encoded.index_pos + i) as u32))
                .collect(),
        };
        let (cos, sin) = rope.rows_host(&positions);
        let half = rope.half();
        let rows = encoded.embeddings.to_vec_f32();
        let hidden = rows.len() / n;
        let mut logits = None;
        for i in 0..n {
            let out = self.embed_step(
                &rows[i * hidden..(i + 1) * hidden],
                &cos[i * half..(i + 1) * half],
                &sin[i * half..(i + 1) * half],
                device,
                cache,
                i + 1 == n,
            );
            self.commit_step(out.as_ref(), device, cache)?;
            logits = out.or(logits);
        }
        logits.ok_or_else(|| fusor::Error::Shape("forward of no tokens".into()))
    }

    /// Resolve one step's outputs and commit the caches, the fixed-cache
    /// step protocol: this step's KV writes (always) plus the logits on the
    /// sampled step, then every cache adopts its written buffer so the
    /// *same* graph runs the next step.
    ///
    /// The batch is the one genuinely rank-heterogeneous list here: a
    /// `[1, vocab]` logits row beside `[1, kv_heads, len, dim]` cache
    /// writes. That is what `resolve` takes and why the caches hand their
    /// pending roots over as `Dyn`.
    fn commit_step(
        &self,
        out: Option<&Tensor<2>>,
        device: &Device,
        cache: &mut LlamaCache,
    ) -> Result<()> {
        let mut batch = Vec::with_capacity(2 * cache.blocks.len() + 1);
        if let Some(out) = out {
            batch.push(out.clone().into_dyn());
        }
        for block in &cache.blocks {
            block.pending_into(&mut batch);
        }
        device.session().resolve(&batch)?;
        for block in &mut cache.blocks {
            block.commit();
        }
        Ok(())
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        images: &[LlamaImage],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2>> {
        if cache
            .as_ref()
            .is_some_and(|c| c.blocks.first().is_some_and(|b| b.is_fixed()))
        {
            let cache = cache.as_deref_mut().expect("checked above");
            if !images.is_empty() {
                return self.forward_fixed_embeds(tokens, images, device, cache);
            }
            return self.forward_fixed_tokens(tokens, device, cache);
        }
        let hidden = self.forward_last_hidden_f32(tokens, images, device, cache)?;
        Ok(self.output.mat_mul(&hidden))
    }

    /// One decode-shaped step: token `token` at absolute `position`, one
    /// query against the (symbolic-length) caches. The graph this builds is
    /// **identical** across steps — same leaves, same nodes — so from step
    /// two on, saturation and extraction are replays and the plan is reused;
    /// only leaf bytes and the length bindings change.
    ///
    /// Because it is identical, it is built once. `cache.decode_graph` holds
    /// the root and every later step re-arms the blocks' appends
    /// ([`fusor::cache::KvCache::replay_append`]) instead of re-deriving the
    /// nodes that produced them. A rebuild is still what happens whenever the
    /// nodes would genuinely differ — a grown store, a reset cache, a first
    /// step — and it re-establishes the memo.
    ///
    /// `None` is a prefill step: its product is the KV writes it left in the
    /// caches, and the head is not run.
    fn decode_step(
        &self,
        token: u32,
        position: usize,
        device: &Device,
        cache: &mut LlamaCache,
        want_logits: bool,
    ) -> Option<Tensor<2>> {
        let (ids, pos) = self.step_inputs.get_or_init(|| {
            (
                Tensor::leaf(device, [Dim::Const(1)]),
                Tensor::leaf(device, [Dim::Const(1)]),
            )
        });
        ids.set_elements(&[token]);
        pos.set_elements(&[position as u32]);

        if let Some(logits) = cache.decode_graph.clone() {
            // Every block or none: a half-advanced cache would silently
            // disagree with the graph about its own length.
            if cache.blocks.iter().all(|block| block.can_replay(1)) {
                for block in &mut cache.blocks {
                    block
                        .replay_append(1)
                        .expect("can_replay was checked for every block");
                }
                return want_logits.then_some(logits);
            }
        }
        // The nodes below may not be the memoized ones (a grown store mints a
        // new leaf), so the memo dies here and is re-established only by a
        // step that actually builds the head. The embedding-row memo dies
        // with it: the appends this step arms are not its.
        cache.decode_graph = None;
        cache.embed_graph = None;

        let mut layer_in = self.tok_embeddings.rows_at(ids).unsqueeze(0);
        if let Some(scale) = self.tok_embedding_scale {
            layer_in = layer_in.mul_scalar(scale);
        }
        let at = RopeAt::Leaf(pos);
        let logits = self.decode_layers(layer_in, at, cache, want_logits);
        if let Some(logits) = &logits {
            cache.decode_graph = Some(logits.clone());
        }
        logits
    }

    /// [`Self::decode_step`] for an embedding row at explicit rope rows: the
    /// step an image prompt's tokens take. Same memo discipline, on
    /// `cache.embed_graph`.
    fn embed_step(
        &self,
        row: &[f32],
        cos: &[f32],
        sin: &[f32],
        device: &Device,
        cache: &mut LlamaCache,
        want_logits: bool,
    ) -> Option<Tensor<2>> {
        let (emb, cos_slot, sin_slot) = self.embed_inputs.get_or_init(|| {
            let hidden = Dim::Const(row.len() as u64);
            let half = Dim::Const(cos.len() as u64);
            (
                Tensor::leaf(device, [Dim::Const(1), Dim::Const(1), hidden]),
                Tensor::leaf(device, [Dim::Const(1), half]),
                Tensor::leaf(device, [Dim::Const(1), half]),
            )
        });
        emb.set_elements(row);
        cos_slot.set_elements(cos);
        sin_slot.set_elements(sin);

        if let Some(logits) = cache.embed_graph.clone() {
            if cache.blocks.iter().all(|block| block.can_replay(1)) {
                for block in &mut cache.blocks {
                    block
                        .replay_append(1)
                        .expect("can_replay was checked for every block");
                }
                return want_logits.then_some(logits);
            }
        }
        cache.embed_graph = None;
        let at = RopeAt::Rows {
            cos: cos_slot,
            sin: sin_slot,
        };
        let logits = self.decode_layers(emb.clone(), at, cache, want_logits);
        if let Some(logits) = &logits {
            cache.embed_graph = Some(logits.clone());
        }
        logits
    }

    /// One `[1, 1, hidden]` embedding through every layer against the
    /// fixed caches, and the head when `want_logits`.
    fn decode_layers(
        &self,
        mut layer_in: Tensor<3>,
        at: RopeAt<'_>,
        cache: &mut LlamaCache,
        want_logits: bool,
    ) -> Option<Tensor<2>> {
        for (i, layer) in self.layers.iter().enumerate() {
            let residual = layer_in.clone();
            let x = layer.attention_norm.forward(&layer_in);
            // One query sees every cached key: structurally maskless.
            let mut attn =
                layer.forward(&x, (MaskKind::None, None), at, Some(&mut cache.blocks[i]));
            if let Some(post_attention_norm) = &layer.post_attention_norm {
                attn = post_attention_norm.forward(&attn);
            }
            let x = layer.ffn_norm.forward_residual(&attn, &residual);
            if layer.post_ffn_norm.is_none() {
                if let Some(layer_out) = layer
                    .feed_forward_variant
                    .forward_add_residuals(&x, &attn, &residual)
                {
                    layer_in = layer_out;
                    continue;
                }
            }
            let mut x = layer.feed_forward_variant.forward(&x);
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                x = post_ffn_norm.forward(&x);
            }
            layer_in = x.add(&attn).add(&residual);
        }
        if !want_logits {
            return None;
        }
        let x = self.norm.forward(&layer_in);
        let hidden = x.reshape_dims([Dim::Const(1), x.extent(2)]);
        Some(self.output.mat_mul(&hidden))
    }

    pub(crate) fn forward_last_hidden_f32(
        &self,
        tokens: &[u32],
        images: &[LlamaImage],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2>> {
        let encoded = self.encode_tokens(tokens, images, device, cache.as_deref_mut())?;
        if encoded.seq_len <= 1 {
            return self.forward_last_hidden_from_embeddings(encoded, device, cache);
        }
        // Chunk the prefill to one token per step. fusor's extraction
        // currently spells an M > 1 activation against a quantized weight as
        // a fold over the *materialized* dequantized matrix — ~27 GB of f32
        // launch roots for one 8B prefill — while M = 1 is the tuned
        // staged-decode contraction that reads the blocks in place. Until
        // batched quantized contraction extraction is fixed, a prefill is a
        // sequence of decode steps.
        let EncodedTokens {
            embeddings,
            seq_len,
            index_pos,
            positions,
        } = encoded;
        let mut last = None;
        for i in 0..seq_len {
            let step = EncodedTokens {
                embeddings: embeddings.narrow(1, i, 1),
                seq_len: 1,
                index_pos: index_pos + i,
                positions: positions.as_ref().map(|p| vec![p[i]]),
            };
            last = Some(self.forward_last_hidden_from_embeddings(
                step,
                device,
                cache.as_deref_mut(),
            )?);
        }
        Ok(last.expect("seq_len > 1 produced at least one step"))
    }

    fn forward_last_hidden_from_embeddings(
        &self,
        encoded: EncodedTokens,
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2>> {
        let EncodedTokens {
            embeddings: mut layer_in,
            seq_len,
            index_pos,
            positions,
        } = encoded;
        let graph = device.graph().clone();
        let tables = tables_for(&self.layers[0].rope_cache, positions.as_deref(), device);
        // Every token of this call shares one placement: a multi-token call
        // is text at consecutive offsets, and an image token arrives alone.
        let mut row = None;
        let at = match (positions.as_deref(), tables.as_ref()) {
            (Some(p), Some((cos, sin))) if seq_len == 1 && p[0].scalar().is_none() => {
                row = Some((cos.clone(), sin.clone()));
                let (cos, sin) = row.as_ref().expect("just set");
                RopeAt::Rows { cos, sin }
            }
            (Some(p), _) => RopeAt::Offset(p[0].0[0] as usize),
            (None, _) => RopeAt::Offset(index_pos),
        };
        let _ = &row;

        for (i, layer) in self.layers.iter().enumerate() {
            let residual = layer_in.clone();
            let x = layer.attention_norm.forward(&layer_in);
            let cache_block = cache.as_deref_mut().map(|c| &mut c.blocks[i]);
            let k_len = cache_block
                .as_ref()
                .and_then(|c| c.len().as_const())
                .unwrap_or(0) as usize
                + seq_len;
            let (kind, mask) = self.mask_for(&graph, seq_len, k_len, layer.sliding_window_size)?;
            let mut attn = layer.forward(&x, (kind, mask.as_ref()), at, cache_block);
            if let Some(post_attention_norm) = &layer.post_attention_norm {
                attn = post_attention_norm.forward(&attn);
            }

            // MLP over RMSNorm(attention_output + residual). The fused path
            // avoids materializing the mid-block residual add just to feed
            // normalization.
            let x = layer.ffn_norm.forward_residual(&attn, &residual);
            if layer.post_ffn_norm.is_none() {
                if let Some(layer_out) = layer
                    .feed_forward_variant
                    .forward_add_residuals(&x, &attn, &residual)
                {
                    layer_in = layer_out;
                    continue;
                }
            }
            let mut x = layer.feed_forward_variant.forward(&x);
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                x = post_ffn_norm.forward(&x);
            }
            layer_in = x.add(&attn).add(&residual);
        }
        let x = self.norm.forward(&layer_in);
        // The last token's hidden state, as `[1, hidden]`.
        let hidden_size = x.extent(2);
        Ok(x.narrow(1, seq_len - 1, 1)
            .reshape_dims([Dim::Const(1), hidden_size]))
    }

    #[allow(dead_code)]
    pub(crate) fn output_matrix(&self) -> &Weight {
        &self.output
    }
}
