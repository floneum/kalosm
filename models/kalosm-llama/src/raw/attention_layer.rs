use crate::raw::rope::RopeImplementation;

use fusor2::cache::{KvCache, MaskKind};
use fusor2::composite::attention::attention_masked;
use fusor2::layers::RmsNorm;
use fusor2::tensor::Dyn as Tensor;
use fusor2::{QMatrix, Result};
use fusor2::Dim;

const MINUS1: usize = usize::MAX;

/// `t.narrow` with `MINUS1` meaning the last axis.
fn narrow(t: &Tensor, dim: usize, start: usize, len: usize) -> Result<Tensor> {
    let dim = if dim == MINUS1 { t.rank() - 1 } else { dim };
    t.narrow(dim, start, len)
}

/// `x @ w^T` for an activation of any rank: fusor2's contraction wants the
/// batch ranks to match, so the leading axes are flattened into the row axis
/// and restored afterwards.
pub(crate) fn q_mat_mul(w: &fusor2::QMatrix, x: &Tensor) -> Result<Tensor> {
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

pub enum FeedForwardVariant {
    // Used by the Llama, Qwen, and Gemma models
    Llama(Box<LlamaFeedForward>),
    // Used by the Phi models
    Phi(PhiFeedForward),
}

impl FeedForwardVariant {
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward(x),
            FeedForwardVariant::Phi(ffn) => ffn.forward(x),
        }
    }

    pub(crate) fn forward_add_residuals(
        &self,
        x: &Tensor,
        first: &Tensor,
        second: &Tensor,
    ) -> Result<Option<Tensor>> {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward_add_residuals(x, first, second),
            FeedForwardVariant::Phi(_) => Ok(None),
        }
    }
}

pub struct PhiFeedForward {
    pub up: QMatrix,
    pub down: QMatrix,
    pub feed_forward_length: usize,
}

impl PhiFeedForward {
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up_states = q_mat_mul(&self.up, x)?;
        let gate = narrow(&up_states, MINUS1, 0, self.feed_forward_length)?;
        let up_states = narrow(
            &up_states,
            MINUS1,
            self.feed_forward_length,
            self.feed_forward_length,
        )?;
        let gate = gate.silu()?;
        let up_states = up_states.mul(&gate)?;
        q_mat_mul(&self.down, &up_states)
    }
}

pub struct LlamaFeedForward {
    gate: QMatrix,
    gate_up: Option<QMatrix>,
    gate_bias: Option<Tensor>,
    down: QMatrix,
    down_bias: Option<Tensor>,
    up: QMatrix,
    up_bias: Option<Tensor>,
}

impl LlamaFeedForward {
    pub(crate) fn new(gate: QMatrix, down: QMatrix, up: QMatrix) -> Self {
        let gate_up = QMatrix::concat_rows(&[&gate, &up]).ok();
        Self {
            gate,
            gate_up,
            down,
            up,
            gate_bias: None,
            down_bias: None,
            up_bias: None,
        }
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up_result = self.activation(x)?;
        let mut up = q_mat_mul(&self.down, &up_result)?;
        if let Some(ref bias) = self.down_bias {
            up = up.add_(bias)?;
        }
        Ok(up)
    }

    /// The decode form: `down(activation(x)) + first + second`, authored in
    /// natural graph form so the resolver can fold the adds into the qmatmul
    /// epilogue.
    pub(crate) fn forward_add_residuals(
        &self,
        x: &Tensor,
        first: &Tensor,
        second: &Tensor,
    ) -> Result<Option<Tensor>> {
        if self.down_bias.is_some() {
            return Ok(None);
        }
        let up_result = self.activation(x)?;
        let projected = q_mat_mul(&self.down, &up_result)?;
        let with_first = projected.add(first)?;
        Ok(Some(with_first.add(second)?))
    }

    fn activation(&self, x: &Tensor) -> Result<Tensor> {
        match &self.gate_up {
            Some(gate_up) if self.gate_bias.is_none() && self.up_bias.is_none() => {
                // SwiGLU over one fused gate|up projection.
                let pair_len = match gate_up.rows {
                    Dim::Const(rows) => rows as usize / 2,
                    _ => unreachable!("gguf rows are const"),
                };
                let projected = q_mat_mul(gate_up, x)?;
                let gate = narrow(&projected, MINUS1, 0, pair_len)?;
                let up = narrow(&projected, MINUS1, pair_len, pair_len)?;
                gate.silu()?.mul(&up)
            }
            _ => {
                let mut w1 = q_mat_mul(&self.gate, x)?;
                if let Some(ref bias) = self.gate_bias {
                    w1 = w1.add_(bias)?;
                }
                let w1 = w1.silu()?;

                let mut w3 = q_mat_mul(&self.up, x)?;
                if let Some(ref bias) = self.up_bias {
                    w3 = w3.add_(bias)?;
                }

                w1.mul(&w3)
            }
        }
    }
}

pub enum AttentionVariant {
    Separate(Box<SeparateAttention>),
    Grouped(GroupedAttention),
}

pub struct AttentionBias {
    bias_q: Tensor,
    bias_k: Tensor,
    bias_v: Tensor,
    bias_qkv: Tensor,
}

impl AttentionBias {
    pub fn new(q: Tensor, k: Tensor, v: Tensor) -> Result<Self> {
        let bias_qkv = Tensor::cat(&[q.clone(), k.clone(), v.clone()], 0)?;
        Ok(Self {
            bias_q: q,
            bias_k: k,
            bias_v: v,
            bias_qkv,
        })
    }
}

pub struct SeparateAttention {
    pub attention_wq: QMatrix,
    /// The row-concatenated `q|k|v` projection, when the three formats agree.
    pub attention_qkv: Option<QMatrix>,
    pub attention_q_norm: Option<RmsNorm>,
    pub attention_wk: QMatrix,
    pub attention_k_norm: Option<RmsNorm>,
    pub attention_wv: QMatrix,
    pub bias: Option<AttentionBias>,
    pub interleaved_rope: bool,
}

fn split_heads(x: &Tensor, b_sz: usize, seq_len: usize, heads: usize, head_dim: usize) -> Result<Tensor> {
    x.reshape_dims(&[
        Dim::Const(b_sz as u64),
        Dim::Const(seq_len as u64),
        Dim::Const(heads as u64),
        Dim::Const(head_dim as u64),
    ])?
    .transpose(1, 2)
}

impl SeparateAttention {
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        hidden_states: &Tensor,
        rope_cache: &RopeImplementation,
        start_pos: usize,
        positions: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len) = match &hidden_states.shape()[..] {
            [Dim::Const(b), Dim::Const(s), _] => (*b as usize, *s as usize),
            other => unreachable!("attention input is rank 3 with const extents, got {other:?}"),
        };

        let (query_states, key_states, value_states) =
            if let Some(attention_qkv) = &self.attention_qkv {
                let query_width = num_heads * head_dim;
                let key_width = num_key_value_heads * head_dim;
                let value_width = key_width;
                let mut qkv = q_mat_mul(attention_qkv, hidden_states)?;
                if let Some(bias) = &self.bias {
                    qkv = qkv.add_(&bias.bias_qkv)?;
                }
                (
                    narrow(&qkv, MINUS1, 0, query_width)?,
                    narrow(&qkv, MINUS1, query_width, key_width)?,
                    narrow(&qkv, MINUS1, query_width + key_width, value_width)?,
                )
            } else {
                let mut q = q_mat_mul(&self.attention_wq, hidden_states)?;
                let mut k = q_mat_mul(&self.attention_wk, hidden_states)?;
                let mut v = q_mat_mul(&self.attention_wv, hidden_states)?;
                if let Some(bias) = &self.bias {
                    q = q.add_(&bias.bias_q)?;
                    k = k.add_(&bias.bias_k)?;
                    v = v.add_(&bias.bias_v)?;
                }
                (q, k, v)
            };

        let mut query = split_heads(&query_states, b_sz, seq_len, num_heads, head_dim)?;
        if let Some(norm) = &self.attention_q_norm {
            query = norm.forward(&query)?;
        }
        let mut key = split_heads(&key_states, b_sz, seq_len, num_key_value_heads, head_dim)?;
        if let Some(norm) = &self.attention_k_norm {
            key = norm.forward(&key)?;
        }
        let value = split_heads(&value_states, b_sz, seq_len, num_key_value_heads, head_dim)?;

        let (query, key) =
            rope_cache.forward(&query, &key, start_pos, self.interleaved_rope, positions)?;
        Ok((query, key, value))
    }
}

pub struct GroupedAttention {
    pub attention_qkv: QMatrix,
    pub interleaved_rope: bool,
}

impl GroupedAttention {
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        x: &Tensor,
        rope_cache: &RopeImplementation,
        start_pos: usize,
        positions: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, seq_len) = match &x.shape()[..] {
            [Dim::Const(b), Dim::Const(s), _] => (*b as usize, *s as usize),
            other => unreachable!("attention input is rank 3 with const extents, got {other:?}"),
        };
        let qkv = q_mat_mul(&self.attention_qkv, x)?;

        let query_pos = num_heads * head_dim;
        let kv_width = num_key_value_heads * head_dim;
        let query_states = narrow(&qkv, MINUS1, 0, query_pos)?;
        let key_states = narrow(&qkv, MINUS1, query_pos, kv_width)?;
        let value_states = narrow(&qkv, MINUS1, query_pos + kv_width, kv_width)?;

        let query = split_heads(&query_states, b_sz, seq_len, num_heads, head_dim)?;
        let key = split_heads(&key_states, b_sz, seq_len, num_key_value_heads, head_dim)?;
        let value = split_heads(&value_states, b_sz, seq_len, num_key_value_heads, head_dim)?;

        let (query, key) =
            rope_cache.forward(&query, &key, start_pos, self.interleaved_rope, positions)?;
        Ok((query, key, value))
    }
}

pub struct LlamaAttention {
    pub attention_variant: AttentionVariant,
    pub attention_wo: QMatrix,
    pub attention_norm: RmsNorm,
    pub post_attention_norm: Option<RmsNorm>,
    pub feed_forward_variant: FeedForwardVariant,
    pub ffn_norm: RmsNorm,
    pub post_ffn_norm: Option<RmsNorm>,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub rope_cache: RopeImplementation,
    pub(crate) sliding_window_size: Option<usize>,
}

impl LlamaAttention {
    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        mask: (MaskKind, Option<&Tensor>),
        start_pos: usize,
        positions: Option<&Tensor>,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let (b_sz, q_len) = match &hidden_states.shape()[..] {
            [Dim::Const(b), Dim::Const(s), _] => (*b as usize, *s as usize),
            other => unreachable!("attention input is rank 3 with const extents, got {other:?}"),
        };
        let hidden_size = self.hidden_size;
        let num_heads = self.n_head;
        let head_dim = self.head_dim;
        let num_key_value_heads = self.n_kv_head;

        let (query_states, key_states, value_states) = match self.attention_variant {
            AttentionVariant::Separate(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
                positions,
            )?,
            AttentionVariant::Grouped(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
                positions,
            )?,
        };

        let mut cache = cache;
        let (key_states, value_states) = match cache.as_deref_mut() {
            None => (key_states, value_states),
            Some(cache) if cache.is_fixed() => {
                // Fixed mode: the append is a scatter into a persistent
                // buffer; a windowed layer is a ring, so eviction is the
                // write itself and no keep_last runs.
                cache.append(&key_states, &value_states)?
            }
            Some(cache) => {
                // The first append stores the value itself, and ours is a
                // transpose/narrow *view* of the projection — a pure view
                // cannot be materialized as a resolve root, which the
                // post-step detach needs it to be. `mul_scalar(1.0)` mints a
                // map member the extractor can land in a buffer.
                let (key_states, value_states) = if cache.k.is_empty() {
                    (key_states.mul_scalar(1.0)?, value_states.mul_scalar(1.0)?)
                } else {
                    (key_states, value_states)
                };
                let (k, v) = cache.append(&key_states, &value_states)?;
                // Sliding-window layers keep only the newest `window` keys:
                // on decode (`q_len == 1`) evicting *before* attention leaves
                // exactly the keys the window admits, so no mask is needed.
                if let (Some(window), 1) = (self.sliding_window_size, q_len) {
                    let k = cache.k.keep_last(window as u64)?.unwrap_or(k);
                    let v = cache.v.keep_last(window as u64)?.unwrap_or(v);
                    (k, v)
                } else {
                    (k, v)
                }
            }
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let (kind, mask_tensor) = mask;
        let attn_output = attention_masked(
            &query_states,
            &key_states,
            &value_states,
            kind,
            mask_tensor,
            Some(scale),
        )?;

        // A prefill on a sliding-window layer evicts after attention: the
        // materialized mask already bounded what each query saw.
        if q_len > 1 {
            if let (Some(window), Some(cache)) = (self.sliding_window_size, cache.as_deref_mut()) {
                if !cache.is_fixed() {
                    cache.k.keep_last(window as u64)?;
                    cache.v.keep_last(window as u64)?;
                }
            }
        }

        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output = attn_output.reshape_dims(&[
            Dim::Const(b_sz as u64),
            Dim::Const(q_len as u64),
            Dim::Const(hidden_size as u64),
        ])?;

        q_mat_mul(&self.attention_wo, &attn_output)
    }
}
