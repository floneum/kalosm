// Adapted from an upstream Whisper quantized model implementation, ported to
// fusor2's dynamic-rank `Tensor`.

use std::num::NonZeroUsize;

use fusor2::cache::{AttentionMask, KvCache, MaskCache, TensorCache};
use fusor2::device::Device;
use fusor2::layers::{ConvNd, LayerNorm, Linear};
use fusor2::tensor::Dyn as Tensor;
use fusor2::{Dim, Dtype, Error, QMatrix, Result, VarBuilder};
use fusor2_gguf::RawTensorBytes;
use timestamps::extract_timestamps;

use crate::config::Config;

pub(crate) mod timestamps;

/// The `[rows, cols]` quantized matrix a GGUF tensor denotes. `fusor2_gguf`
/// already reverses GGUF's fastest-varying-first dims at read, so `raw.shape`
/// is row-major as-is.
fn qmatrix_from_raw(graph: &fusor2::Graph, raw: &RawTensorBytes) -> Result<QMatrix> {
    let Dtype::Q(fmt) = raw.fmt else {
        return Err(Error::Dtype(format!(
            "{} has dtype {:?}; expected a block-quantized matrix",
            raw.name, raw.fmt
        )));
    };
    let (rows, cols) = match raw.shape.as_slice() {
        [cols] => (1, *cols),
        [rows, cols] => (*rows, *cols),
        other => {
            return Err(Error::Shape(format!(
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

/// A GGUF entry as a dense f32 tensor of its stored (row-major) shape.
fn dense(graph: &fusor2::Graph, raw: &RawTensorBytes) -> Result<Tensor> {
    let shape: Vec<Dim> = raw.shape.iter().map(|d| Dim::Const(*d)).collect();
    match raw.fmt {
        Dtype::F32 => Tensor::from_slice(graph.handle(), Dtype::F32, &shape, &raw.bytes),
        Dtype::F16 | Dtype::BF16 => {
            Tensor::from_slice(graph.handle(), raw.fmt, &shape, &raw.bytes)?.cast(Dtype::F32)
        }
        Dtype::Q(_) => qmatrix_from_raw(graph, raw)?
            .dequantize()?
            .reshape_dims(&shape),
        other => Err(Error::Dtype(format!(
            "{} has dtype {other:?}, which has no dense path",
            raw.name
        ))),
    }
}

/// `x @ w^T` for an activation of any rank: fusor2's contraction wants the
/// batch ranks to match, so leading axes are flattened into the row axis and
/// restored afterwards.
fn q_mat_mul(w: &QMatrix, x: &Tensor) -> Result<Tensor> {
    if x.rank() <= 2 {
        return w.q_mat_mul(x);
    }
    let shape = x.shape();
    let lead: Vec<Dim> = shape[..shape.len() - 1].to_vec();
    let k = shape[shape.len() - 1];
    let rows: u64 = lead
        .iter()
        .map(|d| d.as_const().expect("activation extents are constant"))
        .product();
    let flat = x.reshape_dims(&[Dim::Const(rows), k])?;
    let y = w.q_mat_mul(&flat)?;
    let mut out = lead;
    out.push(w.rows);
    y.reshape_dims(&out)
}

fn conv1d(
    padding: u32,
    stride: u32,
    graph: &fusor2::Graph,
    vb: &VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
) -> Result<ConvNd> {
    let weight = dense(graph, &vb.get_raw("weight")?)?;
    // Handle both 2D and 3D weight formats.
    let weight = if weight.rank() == 3 {
        weight
    } else {
        weight.reshape_dims(&[
            Dim::Const(out_channels as u64),
            Dim::Const(in_channels as u64),
            Dim::Const(kernel_size as u64),
        ])?
    };
    let bias = dense(graph, &vb.get_raw("bias")?)?;
    let bias = match bias.rank() {
        1 => bias,
        2 if bias.dim(0).known_eq(Dim::Const(1)) => bias.squeeze(0)?,
        2 => bias.squeeze(1)?,
        r => {
            return Err(Error::Shape(format!(
                "conv bias has rank {r}; expected a vector"
            )));
        }
    };
    Ok(ConvNd::with_config(
        weight,
        Some(bias),
        &[stride],
        &[padding],
        1,
    ))
}

fn load_linear(vb: &VarBuilder, graph: &fusor2::Graph) -> Result<Linear> {
    Linear::load(vb, graph.handle(), vb.contains_key("bias"))
}

struct MultiHeadAttentionCache {
    kv_cache: KvCache,
}

impl MultiHeadAttentionCache {
    fn new() -> Self {
        Self {
            // Keys/values are `[batch, seq, n_state]`; the cache grows along
            // the sequence axis.
            kv_cache: KvCache::new(1),
        }
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
}

impl MultiHeadAttention {
    fn load(n_head: usize, graph: &fusor2::Graph, vb: &VarBuilder) -> Result<Self> {
        let query = load_linear(&vb.pp("q_proj"), graph)?;
        let value = load_linear(&vb.pp("v_proj"), graph)?;
        let key = load_linear(&vb.pp("k_proj"), graph)?;
        let out = load_linear(&vb.pp("out_proj"), graph)?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
        })
    }

    fn forward_kv(
        &self,
        x: &Tensor,
        cache: Option<&mut MultiHeadAttentionCache>,
    ) -> Result<(Tensor, Tensor)> {
        let key_states = self.key.forward(x)?;
        let value_states = self.value.forward(x)?;
        match cache {
            None => Ok((key_states, value_states)),
            Some(cache) => cache.kv_cache.append(&key_states, &value_states),
        }
    }

    fn forward(
        &self,
        query: &Tensor,
        kv: (Tensor, Tensor),
        mask: Option<&AttentionMask>,
        attention_output: Option<&mut TensorCache>,
    ) -> Result<Tensor> {
        let query_states = self.query.forward(query)?;
        let (key_states, value_states) = &kv;
        let wv = self.qkv_attention(
            &query_states,
            key_states,
            value_states,
            mask,
            attention_output,
        )?;
        self.out.forward(&wv)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&AttentionMask>,
        attention_output: Option<&mut TensorCache>,
    ) -> Result<Tensor> {
        let (n_batch, n_ctx_q, n_state) = (q.dim(0), q.dim(1), q.dim(2));
        let n_ctx_kv = k.dim(1);
        let n_state_c = n_state
            .as_const()
            .ok_or_else(|| Error::Shape("attention width must be constant".into()))? as usize;
        let head_dim = n_state_c / self.n_head;
        let scale = (head_dim as f32).powf(-0.25);
        let heads = Dim::Const(self.n_head as u64);
        let hd = Dim::Const(head_dim as u64);

        // [b, s, n_state] -> [b, n_head, s, head_dim]
        let q = q
            .reshape_dims(&[n_batch, n_ctx_q, heads, hd])?
            .transpose(1, 2)?
            .mul_scalar(scale)?;
        let k = k
            .reshape_dims(&[n_batch, n_ctx_kv, heads, hd])?
            .transpose(1, 2)?
            .transpose(2, 3)?
            .mul_scalar(scale)?;
        let v = v
            .reshape_dims(&[n_batch, n_ctx_kv, heads, hd])?
            .transpose(1, 2)?;

        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            qk = mask.apply(&qk)?;
        }
        if let Some(out) = attention_output {
            out.append(&qk)?;
        }
        let w = qk.softmax_last_dim()?;
        let wv = w.matmul(&v)?;
        wv.transpose(1, 2)?
            .reshape_dims(&[n_batch, n_ctx_q, n_state])
    }
}

struct ResidualAttentionBlockCache {
    attn: MultiHeadAttentionCache,
    feature_attn_cache: Option<(Tensor, Tensor)>,
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn load(
        n_head: usize,
        cross_attn: bool,
        graph: &fusor2::Graph,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let attn = MultiHeadAttention::load(n_head, graph, &vb.pp("self_attn"))?;
        let attn_ln = LayerNorm::load(&vb.pp("self_attn_layer_norm"), graph.handle(), 1e-5)?;
        let cross_attn = if cross_attn {
            let cross_attn = MultiHeadAttention::load(n_head, graph, &vb.pp("encoder_attn"))?;
            let cross_attn_ln =
                LayerNorm::load(&vb.pp("encoder_attn_layer_norm"), graph.handle(), 1e-5)?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let mlp_linear1 = load_linear(&vb.pp("fc1"), graph)?;
        let mlp_linear2 = load_linear(&vb.pp("fc2"), graph)?;
        let mlp_ln = LayerNorm::load(&vb.pp("final_layer_norm"), graph.handle(), 1e-5)?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
        })
    }

    fn forward(
        &self,
        audio_features_kv: Option<(Tensor, Tensor)>,
        x: &Tensor,
        mask: Option<&AttentionMask>,
        mut cache: Option<&mut ResidualAttentionBlockCache>,
        attention_output: Option<&mut TensorCache>,
    ) -> Result<Tensor> {
        let attn_ln_x = self.attn_ln.forward(x)?;
        let kv = self
            .attn
            .forward_kv(&attn_ln_x, cache.as_mut().map(|cache| &mut cache.attn))?;
        let attn = self.attn.forward(&attn_ln_x, kv, mask, None)?;
        let mut x = x.add(&attn)?;

        if let (Some(kv), Some((attn, ln))) = (audio_features_kv, &self.cross_attn) {
            let ln_x = ln.forward(&x)?;
            let attn_out = attn.forward(&ln_x, kv, None, attention_output)?;
            x = x.add(&attn_out)?;
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        x.add(&mlp)
    }
}

/// The `[length, channels]` sinusoidal positional table, computed on the host
/// exactly as the reference does.
fn sinusoids(length: usize, channels: usize, graph: &fusor2::Graph) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<f32> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let mut data = vec![0f32; length * channels];
    for (t, row) in data.chunks_mut(channels).enumerate() {
        for (j, inv) in inv_timescales.iter().enumerate() {
            let scaled_time = t as f32 * inv;
            row[j] = scaled_time.sin();
            row[channels / 2 + j] = scaled_time.cos();
        }
    }
    Tensor::from_elements(
        graph.handle(),
        &[Dim::Const(length as u64), Dim::Const(channels as u64)],
        &data,
    )
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
pub struct AudioEncoder {
    conv1: ConvNd,
    conv2: ConvNd,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    fn load(graph: &fusor2::Graph, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let n_mels = cfg.num_mel_bins;
        let conv1 = conv1d(1, 1, graph, &vb.pp("conv1"), n_mels, n_state, 3)?;
        let conv2 = conv1d(1, 2, graph, &vb.pp("conv2"), n_state, n_state, 3)?;
        let positional_embedding = sinusoids(n_ctx, n_state, graph)?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| ResidualAttentionBlock::load(n_head, false, graph, &vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let ln_post = LayerNorm::load(&vb.pp("layer_norm"), graph.handle(), 1e-5)?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = x.transpose(1, 2)?;
        let seq_len = x
            .dim(1)
            .as_const()
            .ok_or_else(|| Error::Shape("encoder sequence length must be constant".into()))?
            as usize;

        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.add_(&positional_embedding)?;

        for block in self.blocks.iter() {
            x = block.forward(None, &x, None, None, None)?;
        }
        self.ln_post.forward(&x)
    }
}

#[derive(Default)]
pub struct TextDecoderCache {
    tokens: Vec<u32>,
    blocks: Vec<ResidualAttentionBlockCache>,
    /// Whether the per-block cross-attention K/V have been re-leafed yet.
    cross_detached: bool,
}

impl TextDecoderCache {
    pub fn new() -> Self {
        Self::default()
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L176
pub struct TextDecoder {
    token_embedding: QMatrix,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    max_target_positions: usize,
    mask_cache: std::sync::Mutex<MaskCache>,
    device: Device,
}

impl TextDecoder {
    fn load(device: &Device, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let graph = device.graph();
        let n_head = cfg.decoder_attention_heads;
        let max_target_positions = cfg.max_target_positions;
        let token_embedding = qmatrix_from_raw(graph, &vb.get_raw("embed_tokens.weight")?)?;
        let positional_embedding = dense(graph, &vb.get_raw("embed_positions.weight")?)?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| ResidualAttentionBlock::load(n_head, true, graph, &vb.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let ln = LayerNorm::load(&vb.pp("layer_norm"), graph.handle(), 1e-5)?;
        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            max_target_positions,
            mask_cache: Default::default(),
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        audio_features: &Tensor,
        cache: &mut TextDecoderCache,
        mut attention_output: Option<&mut [TensorCache]>,
    ) -> Result<Tensor> {
        let index_pos = cache.tokens.len();
        cache.tokens.extend_from_slice(tokens);
        let seq_len = tokens.len();
        if index_pos + seq_len > self.max_target_positions {
            return Err(Error::Shape("exceeded max sequence length".to_string()));
        }
        let graph = self.device.graph();
        // One query sees the whole warm cache; more need the rectangular
        // causal mask at this offset.
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(self.mask_cache.lock().unwrap().materialized(
                graph,
                Dim::Const(seq_len as u64),
                Dim::Const((index_pos + seq_len) as u64),
                None,
            )?)
        };

        let ids = Tensor::from_elements(
            graph.handle(),
            &[Dim::Const(seq_len as u64)],
            tokens,
        )?;
        // The model expects a batch dim but this inference loop does not
        // handle it so we add it at this point.
        let token_embedding = self.token_embedding.index_select_rows(&ids)?.unsqueeze(0)?;
        let positional_embedding = self.positional_embedding.narrow(0, index_pos, seq_len)?;
        let mut x = token_embedding.add_(&positional_embedding)?;

        // Add batch dimension to audio_features for forward_kv.
        let audio_features_batched = audio_features.unsqueeze(0)?;

        for (i, block) in self.blocks.iter().enumerate() {
            if cache.blocks.len() <= i {
                cache.blocks.push(ResidualAttentionBlockCache {
                    attn: MultiHeadAttentionCache::new(),
                    feature_attn_cache: block
                        .cross_attn
                        .as_ref()
                        .and_then(|(attn, _)| attn.forward_kv(&audio_features_batched, None).ok()),
                });
            }
            let block_cache = &mut cache.blocks[i];
            let query = block_cache.feature_attn_cache.clone();
            let attention_output = attention_output.as_mut().map(|outputs| &mut outputs[i]);
            x = block.forward(query, &x, mask.as_ref(), Some(block_cache), attention_output)?;
        }

        self.ln.forward(&x)
    }

    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        q_mat_mul(&self.token_embedding, x)
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
    pub(crate) device: Device,
}

impl Whisper {
    pub fn load(device: &Device, vb: &VarBuilder, config: Config) -> Result<Self> {
        let graph = device.graph();
        let encoder = AudioEncoder::load(graph, &vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(device, &vb.pp("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
            device: device.clone(),
        })
    }

    /// Re-leaf every cache after a decode step so the next step's graph does
    /// not chain back through the whole generation history. Resolving the
    /// batch first makes the per-tensor detach readbacks pure downloads.
    pub(crate) fn detach_caches(
        &self,
        cache: &mut TextDecoderCache,
        mut attention_output: Option<&mut [TensorCache]>,
    ) -> Result<()> {
        let mut batch: Vec<Tensor> = Vec::new();
        for block in &cache.blocks {
            if let Some(k) = block.attn.kv_cache.k.current() {
                batch.push(k.clone());
            }
            if let Some(v) = block.attn.kv_cache.v.current() {
                batch.push(v.clone());
            }
            if !cache.cross_detached {
                if let Some((k, v)) = &block.feature_attn_cache {
                    batch.push(k.clone());
                    batch.push(v.clone());
                }
            }
        }
        if let Some(outputs) = attention_output.as_deref() {
            for out in outputs {
                if let Some(t) = out.current() {
                    batch.push(t.clone());
                }
            }
        }
        if batch.is_empty() {
            return Ok(());
        }
        self.device.session().resolve(&batch)?;
        for block in &mut cache.blocks {
            if let Some(k) = block.attn.kv_cache.k.current().cloned() {
                block.attn.kv_cache.k.data = Some(k.detach()?);
            }
            if let Some(v) = block.attn.kv_cache.v.current().cloned() {
                block.attn.kv_cache.v.data = Some(v.detach()?);
            }
            if !cache.cross_detached {
                if let Some((k, v)) = block.feature_attn_cache.clone() {
                    block.feature_attn_cache = Some((k.detach()?, v.detach()?));
                }
            }
        }
        if let Some(outputs) = attention_output.as_deref_mut() {
            for out in outputs {
                if let Some(t) = out.current().cloned() {
                    out.data = Some(t.detach()?);
                }
            }
        }
        cache.cross_detached = true;
        Ok(())
    }

    pub(crate) async fn dtw_timestamps(
        attention_heads: Option<&'static [[usize; 2]]>,
        filter_width: NonZeroUsize,
        n_frames: usize,
        mask: Vec<Vec<bool>>,
        attention_output: &[TensorCache],
    ) -> Result<Vec<Vec<f32>>> {
        let Some(attention_heads) = attention_heads else {
            panic!("The attention heads for word-level timestamps are not available for this model");
        };

        let mut attention_output_tensor = Vec::new();
        for attn in attention_output {
            attention_output_tensor.push(
                attn.current()
                    .ok_or_else(|| Error::Shape("empty attention output cache".into()))?
                    .clone(),
            );
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
