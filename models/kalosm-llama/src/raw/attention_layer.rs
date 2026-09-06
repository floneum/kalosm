use crate::raw::rope::RopeImplementation;

use crate::raw::weight::Weight;
use fusor::cache::{KvCache, MaskKind};
use fusor::layers::RmsNorm;
use fusor::Minus1;
use fusor::Tensor;

pub enum FeedForwardVariant {
    // Used by the Llama, Qwen, and Gemma models
    Llama(Box<LlamaFeedForward>),
    // Used by the Phi models
    Phi(PhiFeedForward),
}

impl FeedForwardVariant {
    pub(crate) fn forward(&self, x: &Tensor<3>) -> Tensor<3> {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward(x),
            FeedForwardVariant::Phi(ffn) => ffn.forward(x),
        }
    }

    pub(crate) fn forward_add_residuals(
        &self,
        x: &Tensor<3>,
        first: &Tensor<3>,
        second: &Tensor<3>,
    ) -> Option<Tensor<3>> {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward_add_residuals(x, first, second),
            FeedForwardVariant::Phi(_) => None,
        }
    }
}

pub struct PhiFeedForward {
    pub up: Weight,
    pub down: Weight,
    pub feed_forward_length: usize,
}

impl PhiFeedForward {
    pub(crate) fn forward(&self, x: &Tensor<3>) -> Tensor<3> {
        let up_states = self.up.mat_mul(x);
        let len = self.feed_forward_length;
        let gate = up_states.narrow(Minus1, 0, len).silu();
        let up_states = up_states.narrow(Minus1, len, len).mul(&gate);
        self.down.mat_mul(&up_states)
    }
}

pub struct LlamaFeedForward {
    gate: Weight,
    gate_up: Option<Weight>,
    gate_bias: Option<Tensor<1>>,
    down: Weight,
    down_bias: Option<Tensor<1>>,
    up: Weight,
    up_bias: Option<Tensor<1>>,
}

impl LlamaFeedForward {
    pub(crate) fn new(gate: Weight, down: Weight, up: Weight) -> Self {
        let gate_up = Weight::concat_rows(&[&gate, &up]);
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

    pub(crate) fn forward(&self, x: &Tensor<3>) -> Tensor<3> {
        let up = self.down.mat_mul(&self.activation(x));
        match &self.down_bias {
            Some(bias) => up.add_(bias),
            None => up,
        }
    }

    /// The decode form: `down(activation(x)) + first + second`, authored in
    /// natural graph form so the resolver can fold the adds into the qmatmul
    /// epilogue.
    pub(crate) fn forward_add_residuals(
        &self,
        x: &Tensor<3>,
        first: &Tensor<3>,
        second: &Tensor<3>,
    ) -> Option<Tensor<3>> {
        if self.down_bias.is_some() {
            return None;
        }
        let projected = self.down.mat_mul(&self.activation(x));
        Some(projected.add(first).add(second))
    }

    fn activation(&self, x: &Tensor<3>) -> Tensor<3> {
        match &self.gate_up {
            Some(gate_up) if self.gate_bias.is_none() && self.up_bias.is_none() => {
                // SwiGLU over one fused gate|up projection.
                let pair_len = gate_up.rows().as_const().expect("gguf rows are const") as usize / 2;
                let projected = gate_up.mat_mul(x);
                let gate = projected.narrow(Minus1, 0, pair_len);
                let up = projected.narrow(Minus1, pair_len, pair_len);
                gate.silu().mul(&up)
            }
            _ => {
                let mut w1 = self.gate.mat_mul(x);
                if let Some(bias) = &self.gate_bias {
                    w1 = w1.add_(bias);
                }
                let w1 = w1.silu();

                let mut w3 = self.up.mat_mul(x);
                if let Some(bias) = &self.up_bias {
                    w3 = w3.add_(bias);
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
    bias_q: Tensor<1>,
    bias_k: Tensor<1>,
    bias_v: Tensor<1>,
    bias_qkv: Tensor<1>,
}

impl AttentionBias {
    pub fn new(q: Tensor<1>, k: Tensor<1>, v: Tensor<1>) -> Self {
        let bias_qkv = Tensor::cat([q.clone(), k.clone(), v.clone()], 0);
        Self {
            bias_q: q,
            bias_k: k,
            bias_v: v,
            bias_qkv,
        }
    }
}

pub struct SeparateAttention {
    pub attention_wq: Weight,
    /// The row-concatenated `q|k|v` projection, when the three formats agree.
    pub attention_qkv: Option<Weight>,
    pub attention_q_norm: Option<RmsNorm>,
    pub attention_wk: Weight,
    pub attention_k_norm: Option<RmsNorm>,
    pub attention_wv: Weight,
    pub bias: Option<AttentionBias>,
    pub interleaved_rope: bool,
}

/// `[batch, seq, heads * head_dim]` seen as `[batch, heads, seq, head_dim]`.
fn split_heads(x: &Tensor<3>, heads: usize, head_dim: usize) -> Tensor<4> {
    let [b_sz, seq_len, _] = x.shape();
    x.reshape([b_sz, seq_len, heads, head_dim]).transpose(1, 2)
}

impl SeparateAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        hidden_states: &Tensor<3>,
        rope_cache: &RopeImplementation,
        start_pos: usize,
        positions: Option<&Tensor<1, u32>>,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let (query_states, key_states, value_states) =
            if let Some(attention_qkv) = &self.attention_qkv {
                let query_width = num_heads * head_dim;
                let key_width = num_key_value_heads * head_dim;
                let value_width = key_width;
                let mut qkv = attention_qkv.mat_mul(&hidden_states);
                if let Some(bias) = &self.bias {
                    qkv = qkv.add_(&bias.bias_qkv);
                }
                (
                    qkv.narrow(Minus1, 0, query_width),
                    qkv.narrow(Minus1, query_width, key_width),
                    qkv.narrow(Minus1, query_width + key_width, value_width),
                )
            } else {
                let mut q = self.attention_wq.mat_mul(&hidden_states);
                let mut k = self.attention_wk.mat_mul(&hidden_states);
                let mut v = self.attention_wv.mat_mul(&hidden_states);
                if let Some(bias) = &self.bias {
                    q = q.add_(&bias.bias_q);
                    k = k.add_(&bias.bias_k);
                    v = v.add_(&bias.bias_v);
                }
                (q, k, v)
            };

        let mut query = split_heads(&query_states, num_heads, head_dim);
        if let Some(norm) = &self.attention_q_norm {
            query = norm.forward(&query);
        }
        let mut key = split_heads(&key_states, num_key_value_heads, head_dim);
        if let Some(norm) = &self.attention_k_norm {
            key = norm.forward(&key);
        }
        let value = split_heads(&value_states, num_key_value_heads, head_dim);

        let (query, key) =
            rope_cache.forward(&query, &key, start_pos, self.interleaved_rope, positions);
        (query, key, value)
    }
}

pub struct GroupedAttention {
    pub attention_qkv: Weight,
    pub interleaved_rope: bool,
}

impl GroupedAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        x: &Tensor<3>,
        rope_cache: &RopeImplementation,
        start_pos: usize,
        positions: Option<&Tensor<1, u32>>,
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let qkv = self.attention_qkv.mat_mul(&x);

        let query_pos = num_heads * head_dim;
        let kv_width = num_key_value_heads * head_dim;
        let query_states = qkv.narrow(Minus1, 0, query_pos);
        let key_states = qkv.narrow(Minus1, query_pos, kv_width);
        let value_states = qkv.narrow(Minus1, query_pos + kv_width, kv_width);

        let query = split_heads(&query_states, num_heads, head_dim);
        let key = split_heads(&key_states, num_key_value_heads, head_dim);
        let value = split_heads(&value_states, num_key_value_heads, head_dim);

        let (query, key) =
            rope_cache.forward(&query, &key, start_pos, self.interleaved_rope, positions);
        (query, key, value)
    }
}

pub struct LlamaAttention {
    pub attention_variant: AttentionVariant,
    pub attention_wo: Weight,
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
        hidden_states: &Tensor<3>,
        mask: (MaskKind, Option<&Tensor<2>>),
        start_pos: usize,
        positions: Option<&Tensor<1, u32>>,
        cache: Option<&mut KvCache>,
    ) -> Tensor<3> {
        let [b_sz, q_len, _] = hidden_states.shape();
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
            ),
            AttentionVariant::Grouped(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
                positions,
            ),
        };

        let mut cache = cache;
        let (key_states, value_states) = match cache.as_deref_mut() {
            None => (key_states, value_states),
            // Fixed mode: the append is a scatter into a persistent buffer; a
            // windowed layer is a ring, so eviction is the write itself and no
            // keep_last runs.
            Some(cache) if cache.is_fixed() => cache.append(&key_states, &value_states),
            Some(cache) => {
                // The first append stores the value itself, and ours is a
                // transpose/narrow *view* of the projection — a pure view
                // cannot be materialized as a resolve root, which the
                // post-step detach needs it to be. `mul_scalar(1.0)` mints a
                // map member the extractor can land in a buffer.
                let (key_states, value_states) = if cache.is_empty() {
                    (key_states.mul_scalar(1.0), value_states.mul_scalar(1.0))
                } else {
                    (key_states, value_states)
                };
                let (k, v) = cache.append(&key_states, &value_states);
                // Sliding-window layers keep only the newest `window` keys:
                // on decode (`q_len == 1`) evicting *before* attention leaves
                // exactly the keys the window admits, so no mask is needed.
                if let (Some(window), 1) = (self.sliding_window_size, q_len) {
                    cache.keep_last(window as u64).unwrap_or((k, v))
                } else {
                    (k, v)
                }
            }
        };

        let scale = 1.0 / (head_dim as f32).sqrt();
        let (kind, mask_tensor) = mask;
        let attn_output = query_states.attention_masked(
            &key_states,
            &value_states,
            kind,
            mask_tensor,
            Some(scale),
        );

        // A prefill on a sliding-window layer evicts after attention: the
        // materialized mask already bounded what each query saw.
        if q_len > 1 {
            if let (Some(window), Some(cache)) = (self.sliding_window_size, cache) {
                if !cache.is_fixed() {
                    cache.keep_last(window as u64);
                }
            }
        }

        self.attention_wo.mat_mul(
            &attn_output
                .transpose(1, 2)
                .reshape([b_sz, q_len, hidden_size]),
        )
    }
}
