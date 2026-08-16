use std::sync::{Arc, Mutex};

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
use fusor::cache::{MaskCache, MaskKind};
use fusor::layers::RmsNorm;
use fusor::{Device, Dim, Dtype, Graph, QMatrix, Result, Tensor};
use fusor_gguf::{GgufValue, RawTensorBytes, ShardedVarBuilder};

mod attention_layer;
pub mod cache;
mod rope;

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
    tok_embeddings: QMatrix,
    tok_embedding_scale: Option<f32>,
    layers: Vec<LlamaAttention>,
    norm: RmsNorm,
    output: QMatrix,
    /// Memoizes the materialized (rectangular / windowed) masks.
    masks: Mutex<MaskCache>,
    /// The decode loop's persistent input leaves: the token id and its
    /// absolute position, both `[1]` `u32`. Only their *bytes* change per
    /// step, so every step reuses one graph and replays one plan.
    step_inputs: std::sync::OnceLock<(Tensor<1, u32>, Tensor<1, u32>)>,
}

/// The embedded token inputs produced by [`Model::encode_tokens`], ready to be
/// run through the transformer layers.
pub(crate) struct EncodedTokens {
    embeddings: Tensor<3>,
    seq_len: usize,
    index_pos: usize,
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

/// The `[rows, cols]` quantized matrix a GGUF tensor denotes.
/// `fusor_gguf` already reverses GGUF's fastest-varying-first dims at read,
/// so `raw.shape` is row-major `[rows, cols]` as-is.
fn qmatrix_from_raw(graph: &Graph, raw: &RawTensorBytes) -> Result<QMatrix> {
    let Dtype::Q(fmt) = raw.fmt else {
        return Err(fusor::Error::Dtype(format!(
            "{} has dtype {:?}; only block-quantized matmul weights are supported by this port",
            raw.name, raw.fmt
        )));
    };
    let (rows, cols) = match raw.shape.as_slice() {
        [cols] => (1, *cols),
        [rows, cols] => (*rows, *cols),
        other => {
            return Err(fusor::Error::Shape(format!(
                "{} has GGUF shape {other:?}; a QMatrix is rank 1 or 2",
                raw.name
            )));
        }
    };
    QMatrix::from_raw_bytes(
        graph,
        fmt,
        raw.layout,
        [Dim::Const(rows), Dim::Const(cols)],
        &raw.bytes,
    )
}

/// A GGUF tensor as a dense rank-1 `f32` value (norm weights, biases,
/// `rope_freqs.weight`).
fn dense_1d(device: &Device, raw: &RawTensorBytes) -> Result<Tensor<1>> {
    let n: u64 = raw.shape.iter().product();
    match raw.fmt {
        // The dtype is data read out of the file; the rank is not.
        Dtype::F32 | Dtype::F16 => Ok(Tensor::from_raw_bytes(
            device,
            raw.fmt,
            [Dim::Const(n)],
            &raw.bytes,
        )),
        Dtype::Q(_) => Ok(qmatrix_from_raw(device.graph(), raw)?
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

        if source.get(".rope.dimension_sections").is_ok() {
            return Err(LlamaSourceError::MissingGgufEntry(
                "mrope (Qwen-VL) models are not supported by the fusor port yet".to_string(),
            ));
        }

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
        };
        let config = Arc::new(config);

        let rope = RopeImplementation::new(&config, config.rope_theta, device);
        let sliding_rope = rope_freq_base_sliding.map(|rope_freq_base_sliding| {
            RopeImplementation::new(&config, rope_freq_base_sliding, device)
        });

        let tok_embeddings_q = qmatrix_from_raw(&graph, &source.tensor("token_embd.weight")?)?;
        let tok_embedding_scale =
            (&*architecture == "gemma3").then(|| (embedding_length as f32).sqrt());

        let norm = source.tensor("output_norm.weight")?;
        let norm = decode_norm(norm, rms_norm_eps)?;
        let output = match source.tensor("output.weight") {
            Ok(output) => qmatrix_from_raw(&graph, &output)?,
            // If there is no output layer, assume the word embeddings are tied to the output
            Err(_) => tok_embeddings_q.clone(),
        };
        let mut layers = Vec::with_capacity(block_count);
        let interleaved_rope = architecture.as_str() != "qwen2"
            && architecture.as_str() != "qwen3"
            && architecture.as_str() != "gemma3";
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_variant = if let Ok(attention_qkv) =
                source.tensor(&format!("{prefix}.attn_qkv.weight"))
            {
                AttentionVariant::Grouped(GroupedAttention {
                    attention_qkv: qmatrix_from_raw(&graph, &attention_qkv)?,
                    interleaved_rope,
                })
            } else {
                let q =
                    qmatrix_from_raw(&graph, &source.tensor(&format!("{prefix}.attn_q.weight"))?)?;
                let k =
                    qmatrix_from_raw(&graph, &source.tensor(&format!("{prefix}.attn_k.weight"))?)?;
                let v =
                    qmatrix_from_raw(&graph, &source.tensor(&format!("{prefix}.attn_v.weight"))?)?;
                let qkv = QMatrix::concat_rows(&[&q, &k, &v]).ok();
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
            let attention_wo = qmatrix_from_raw(
                &graph,
                &source.tensor(&format!("{prefix}.attn_output.weight"))?,
            )?;
            // Try to read from the up, down and gate weights
            let feed_forward_variant = if let Ok(ffn_gate) =
                source.tensor(&format!("{prefix}.ffn_gate.weight"))
            {
                let feed_forward_w1 = qmatrix_from_raw(&graph, &ffn_gate)?;
                let feed_forward_w2 = qmatrix_from_raw(
                    &graph,
                    &source.tensor(&format!("{prefix}.ffn_down.weight"))?,
                )?;
                let feed_forward_w3 =
                    qmatrix_from_raw(&graph, &source.tensor(&format!("{prefix}.ffn_up.weight"))?)?;
                FeedForwardVariant::Llama(Box::new(LlamaFeedForward::new(
                    feed_forward_w1,
                    feed_forward_w2,
                    feed_forward_w3,
                )))
            } else {
                // Otherwise, try to read from the up, and down weights
                let up =
                    qmatrix_from_raw(&graph, &source.tensor(&format!("{prefix}.ffn_up.weight"))?)?;
                let down = qmatrix_from_raw(
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

        Ok(Self {
            config,
            tok_embeddings: tok_embeddings_q,
            tok_embedding_scale,
            layers,
            norm,
            output,
            masks: Mutex::new(MaskCache::new()),
            step_inputs: std::sync::OnceLock::new(),
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

    pub fn encode_tokens(
        &self,
        raw_tokens: &[u32],
        device: &Device,
        cache: Option<&mut LlamaCache>,
    ) -> EncodedTokens {
        let (tokens, index_pos) = self.plan_tokens(raw_tokens, cache);
        let seq_len = tokens.len();
        let ids = Tensor::from_slice(device, [seq_len], &tokens);
        let mut embeddings = self.tok_embeddings.rows_at(&ids).unsqueeze(0);
        if let Some(scale) = self.tok_embedding_scale {
            embeddings = embeddings.mul_scalar(scale);
        }

        EncodedTokens {
            embeddings,
            seq_len,
            index_pos,
        }
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
    pub fn forward(
        &self,
        tokens: &[u32],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2>> {
        if cache
            .as_ref()
            .is_some_and(|c| c.blocks.first().is_some_and(|b| b.is_fixed()))
        {
            let cache = cache.as_deref_mut().expect("checked above");
            let (steps, index_pos) = self.plan_tokens(tokens, Some(cache));
            let n = steps.len();
            let mut logits = None;
            for (i, tok) in steps.iter().enumerate() {
                let want_logits = i + 1 == n;
                let out = self.decode_step(*tok, index_pos + i, device, cache, want_logits);
                // One resolve per step: this step's KV writes (always) plus
                // the logits on the sampled step. Then every cache adopts its
                // written buffer so the *same* graph runs the next step.
                //
                // The batch is the one genuinely rank-heterogeneous list here:
                // a `[1, vocab]` logits row beside `[1, kv_heads, len, dim]`
                // cache writes. That is what `resolve` takes and why the caches
                // hand their pending roots over as `Dyn`.
                let mut batch = Vec::with_capacity(2 * cache.blocks.len() + 1);
                if let Some(out) = &out {
                    batch.push(out.clone().into_dyn());
                }
                for block in &cache.blocks {
                    block.pending_into(&mut batch);
                }
                device.session().resolve(&batch)?;
                for block in &mut cache.blocks {
                    block.commit();
                }
                logits = out.or(logits);
            }
            return logits.ok_or_else(|| fusor::Error::Shape("forward of no tokens".into()));
        }
        let hidden = self.forward_last_hidden_f32(tokens, device, cache)?;
        Ok(hidden.q_mat_mul(&self.output))
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
        // step that actually builds the head.
        cache.decode_graph = None;

        let mut layer_in = self.tok_embeddings.rows_at(ids).unsqueeze(0);
        if let Some(scale) = self.tok_embedding_scale {
            layer_in = layer_in.mul_scalar(scale);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let residual = layer_in.clone();
            let x = layer.attention_norm.forward(&layer_in);
            // One query sees every cached key: structurally maskless.
            let mut attn = layer.forward(
                &x,
                (MaskKind::None, None),
                position,
                Some(pos),
                Some(&mut cache.blocks[i]),
            );
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
        let logits = hidden.q_mat_mul(&self.output);
        cache.decode_graph = Some(logits.clone());
        Some(logits)
    }

    pub(crate) fn forward_last_hidden_f32(
        &self,
        tokens: &[u32],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2>> {
        let encoded = self.encode_tokens(tokens, device, cache.as_deref_mut());
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
        } = encoded;
        let mut last = None;
        for i in 0..seq_len {
            let step = EncodedTokens {
                embeddings: embeddings.narrow(1, i, 1),
                seq_len: 1,
                index_pos: index_pos + i,
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
        } = encoded;
        let graph = device.graph().clone();

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
            let mut attn = layer.forward(&x, (kind, mask.as_ref()), index_pos, None, cache_block);
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
    pub(crate) fn output_matrix(&self) -> &QMatrix {
        &self.output
    }
}
