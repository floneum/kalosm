use crate::raw::rope::RopeImplementation;

use fusor::cache::AttentionMask;
use fusor::cache::KvCache;
use fusor::layers::Linear;
use fusor::layers::RmsNorm;
use fusor::QMatrix;
use fusor::Tensor;
use fusor::D;
use fusor::{CastTensor, CastTo, FloatDataType, Fusion, SimdElement};

pub enum FeedForwardVariant<F: FloatDataType + SimdElement = f32> {
    // Used by the Llama, Qwen, and Gemma models
    Llama(Box<LlamaFeedForward<F>>),
    // Used by the Phi models
    Phi(PhiFeedForward),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeedForwardActivation {
    Silu,
    Gelu,
}

impl<F: FloatDataType + SimdElement + Default> FeedForwardVariant<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    pub(crate) fn forward<B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
    where
        B: Fusion<3, F>,
    {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward(x),
            FeedForwardVariant::Phi(ffn) => ffn.forward(x),
        }
    }

    pub(crate) fn forward_add_residuals<B, B1, B2>(
        &self,
        x: &Tensor<3, F, B>,
        first: &Tensor<3, f32, B1>,
        second: &Tensor<3, f32, B2>,
    ) -> Option<Tensor<3, F>>
    where
        B: Fusion<3, F>,
        B1: Fusion<3, f32>,
        B2: Fusion<3, f32>,
    {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward_add_residuals(x, first, second),
            FeedForwardVariant::Phi(_) => None,
        }
    }
}

pub struct PhiFeedForward {
    pub up: QMatrix,
    pub down: QMatrix,
    pub feed_forward_length: usize,
}

impl PhiFeedForward {
    pub(crate) fn forward<F, B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
    where
        F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
        f32: CastTo<F> + CastTensor<F>,
        B: Fusion<3, F>,
    {
        // All computation happens in f32 for compatibility with SIMD ops
        let x_f32 = x.cast::<f32>();
        let up_states = x_f32.q_mat_mul(&self.up);
        let gate = up_states
            .narrow(D::Minus1, 0, self.feed_forward_length)
            .to_concrete();
        let up_states = up_states
            .narrow(
                D::Minus1,
                self.feed_forward_length,
                self.feed_forward_length,
            )
            .to_concrete();
        let gate = gate.silu();
        let up_states = up_states * gate;
        let result = up_states.q_mat_mul(&self.down);
        result.cast()
    }
}

pub struct LlamaFeedForward<F: FloatDataType + SimdElement = f32> {
    activation: FeedForwardActivation,
    gate: QMatrix,
    gate_up: Option<QMatrix>,
    gate_bias: Option<Tensor<1, F>>,
    down: QMatrix,
    down_bias: Option<Tensor<1, F>>,
    up: QMatrix,
    up_bias: Option<Tensor<1, F>>,
}

impl<F: FloatDataType + SimdElement> LlamaFeedForward<F> {
    pub(crate) fn new_with_activation(
        gate: QMatrix,
        down: QMatrix,
        up: QMatrix,
        activation: FeedForwardActivation,
    ) -> Self {
        let gate_up = QMatrix::concat_rows(&[&gate, &up]);
        Self {
            activation,
            gate,
            gate_up,
            down,
            up,
            gate_bias: None,
            down_bias: None,
            up_bias: None,
        }
    }

    #[cfg(feature = "vision")]
    pub(crate) fn new_with_bias(
        gate: QMatrix,
        gate_bias: Option<Tensor<1, F>>,
        down: QMatrix,
        down_bias: Option<Tensor<1, F>>,
        up: QMatrix,
        up_bias: Option<Tensor<1, F>>,
    ) -> Self {
        let gate_up = QMatrix::concat_rows(&[&gate, &up]);
        Self {
            activation: FeedForwardActivation::Silu,
            gate,
            gate_up,
            gate_bias,
            down,
            down_bias,
            up,
            up_bias,
        }
    }

    pub(crate) fn forward<B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
    where
        F: CastTo<f32> + CastTensor<f32>,
        f32: CastTo<F> + CastTensor<F>,
        B: Fusion<3, F>,
    {
        let up_result = self.activation(x);
        let mut up = up_result.q_mat_mul(&self.down);
        if let Some(ref bias) = self.down_bias {
            let bias_f32: Tensor<1, f32> = bias.cast();
            up = up.add_(&bias_f32);
        }

        // Cast back to F
        up.cast()
    }

    pub(crate) fn forward_add_residuals<B, B1, B2>(
        &self,
        x: &Tensor<3, F, B>,
        first: &Tensor<3, f32, B1>,
        second: &Tensor<3, f32, B2>,
    ) -> Option<Tensor<3, F>>
    where
        F: CastTo<f32> + CastTensor<f32>,
        f32: CastTo<F> + CastTensor<F>,
        B: Fusion<3, F>,
        B1: Fusion<3, f32>,
        B2: Fusion<3, f32>,
    {
        if self.down_bias.is_some() {
            return None;
        }
        if x.shape()[1] > 1 {
            return None;
        }

        let up_result = self.activation(x);
        // Residual adds authored in natural graph form: the resolver folds both
        // `add`s into the qmatmul post epilogue (one dispatch on decode).
        let projected = up_result.q_mat_mul(&self.down);
        let with_first = (&projected + first).to_concrete();
        let up = (&with_first + second).to_concrete();
        Some(up.cast())
    }

    fn activation<B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, f32>
    where
        F: CastTo<f32> + CastTensor<f32>,
        B: Fusion<3, F>,
    {
        // All computation happens in f32 for compatibility with SIMD ops
        let x_f32 = x.cast::<f32>();

        match &self.gate_up {
            Some(gate_up) if self.gate_bias.is_none() && self.up_bias.is_none() => {
                // SwiGLU split/gate authored in natural graph form: the resolver
                // folds `silu(gate) * up` over the two narrow halves into the
                // qmatmul accumulator-offset epilogue (one dispatch on decode).
                let pair_len = gate_up.shape()[0] / 2;
                let projected = x_f32.q_mat_mul(gate_up);
                let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
                let up = projected
                    .narrow(D::Minus1, pair_len, pair_len)
                    .to_concrete();
                (self.activate(gate) * up).to_concrete()
            }
            Some(gate_up) => {
                let gate_width = self.gate.shape()[0];
                let up_width = self.up.shape()[0];
                let gate_up_states = x_f32.q_mat_mul(gate_up);

                let mut gate_states = gate_up_states
                    .narrow(D::Minus1, 0, gate_width)
                    .to_concrete();
                if let Some(ref bias) = self.gate_bias {
                    let bias_f32: Tensor<1, f32> = bias.cast();
                    gate_states = gate_states.add_(&bias_f32);
                }

                let mut up_states = gate_up_states
                    .narrow(D::Minus1, gate_width, up_width)
                    .to_concrete();
                if let Some(ref bias) = self.up_bias {
                    let bias_f32: Tensor<1, f32> = bias.cast();
                    up_states = up_states.add_(&bias_f32);
                }

                (self.activate(gate_states) * up_states).to_concrete()
            }
            None => {
                let mut w1 = x_f32.q_mat_mul(&self.gate);
                if let Some(ref bias) = self.gate_bias {
                    let bias_f32: Tensor<1, f32> = bias.cast();
                    w1 = w1.add_(&bias_f32);
                }
                let w1 = self.activate(w1);

                let mut w3 = x_f32.q_mat_mul(&self.up);
                if let Some(ref bias) = self.up_bias {
                    let bias_f32: Tensor<1, f32> = bias.cast();
                    w3 = w3.add_(&bias_f32);
                }

                (w1 * w3).to_concrete()
            }
        }
    }

    fn activate(&self, x: Tensor<3, f32>) -> Tensor<3, f32> {
        match self.activation {
            FeedForwardActivation::Silu => x.silu(),
            FeedForwardActivation::Gelu => x.gelu(),
        }
    }
}

pub enum AttentionVariant<F: FloatDataType + SimdElement = f32> {
    Separate(Box<SeparateAttention<F>>),
    Grouped(GroupedAttention),
}

pub struct AttentionBias<F: FloatDataType + SimdElement = f32> {
    bias_q: Tensor<1, F>,
    bias_k: Tensor<1, F>,
    bias_v: Tensor<1, F>,
    bias_qkv: Tensor<1, F>,
}

impl<F: FloatDataType + SimdElement + Default> AttentionBias<F> {
    pub fn new(q: Tensor<1, F>, k: Tensor<1, F>, v: Tensor<1, F>) -> Self {
        let bias_qkv = fusor::cat([q.clone(), k.clone(), v.clone()], 0).to_concrete();
        Self {
            bias_q: q,
            bias_k: k,
            bias_v: v,
            bias_qkv,
        }
    }
}

pub struct SeparateAttention<F: FloatDataType + SimdElement = f32> {
    pub attention_wq: QMatrix,
    pub attention_qkv: Option<QMatrix>,
    pub attention_q_norm: Option<RmsNorm<1, F>>,
    pub attention_wk: Option<QMatrix>,
    pub attention_k_norm: Option<RmsNorm<1, F>>,
    pub attention_wv: Option<QMatrix>,
    pub attention_v_norm: Option<RmsNorm<1, F>>,
    pub bias: Option<AttentionBias<F>>,
    pub interleaved_rope: bool,
}

impl<F: FloatDataType + SimdElement + Default> SeparateAttention<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    #[allow(clippy::too_many_arguments)]
    fn forward<B>(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        hidden_states: &Tensor<3, F, B>,
        rope_cache: &RopeImplementation<F>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> (Tensor<4, F>, Tensor<4, F>, Tensor<4, F>)
    where
        B: Fusion<3, F>,
    {
        let [b_sz, seq_len, _] = hidden_states.shape();

        // Compute in f32 for SIMD ops compatibility
        let hidden_f32 = hidden_states.cast::<f32>();

        if let Some(attention_qkv) = &self.attention_qkv {
            let query_width = num_heads * head_dim;
            let key_width = num_key_value_heads * head_dim;
            let value_width = key_width;
            let mut qkv = hidden_f32.q_mat_mul(attention_qkv);
            if let Some(bias) = &self.bias {
                let bias_f32: Tensor<1, f32> = bias.bias_qkv.cast();
                qkv = qkv.add_(&bias_f32);
            }

            let query_states: Tensor<4, F> = {
                let query_states = qkv.narrow(D::Minus1, 0, query_width).to_concrete();

                let query = query_states
                    .reshape([b_sz, seq_len, num_heads, head_dim])
                    .transpose(1, 2)
                    .to_concrete();

                let query: Tensor<4, F> = query.cast();
                if let Some(norm) = &self.attention_q_norm {
                    norm.forward_generic_4d(&query)
                } else {
                    query
                }
            };

            let key_states: Tensor<4, F> = {
                let key_states = qkv.narrow(D::Minus1, query_width, key_width).to_concrete();

                let key = key_states
                    .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
                    .transpose(1, 2)
                    .to_concrete();

                let key: Tensor<4, F> = key.cast();
                if let Some(norm) = &self.attention_k_norm {
                    norm.forward_generic_4d(&key)
                } else {
                    key
                }
            };

            let value_states: Tensor<4, F> = {
                let value_states = qkv
                    .narrow(D::Minus1, query_width + key_width, value_width)
                    .to_concrete();

                let value_states: Tensor<4, F> = value_states
                    .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
                    .transpose(1, 2)
                    .to_concrete()
                    .cast();
                if let Some(norm) = &self.attention_v_norm {
                    norm.forward_generic_4d(&value_states)
                } else {
                    value_states
                }
            };

            let (query_states, key_states) = rope_cache.forward(
                &query_states,
                &key_states,
                start_pos,
                pos_ids,
                self.interleaved_rope,
            );
            return (query_states, key_states, value_states);
        }

        let query_states: Tensor<4, F> = {
            let mut query_states = hidden_f32.q_mat_mul(&self.attention_wq);

            if let Some(bias) = &self.bias {
                let bias_f32: Tensor<1, f32> = bias.bias_q.cast();
                query_states = query_states.add_(&bias_f32);
            }

            let query = query_states
                .reshape([b_sz, seq_len, num_heads, head_dim])
                .transpose(1, 2)
                .to_concrete();

            let query: Tensor<4, F> = query.cast();
            if let Some(norm) = &self.attention_q_norm {
                norm.forward_generic_4d(&query)
            } else {
                query
            }
        };
        let key_states: Tensor<4, F> = {
            let attention_wk = self
                .attention_wk
                .as_ref()
                .expect("separate attention without K weights must use a shared KV cache");
            let mut key_states = hidden_f32.q_mat_mul(attention_wk);

            if let Some(bias) = &self.bias {
                let bias_f32: Tensor<1, f32> = bias.bias_k.cast();
                key_states = key_states.add_(&bias_f32);
            }

            let key = key_states
                .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
                .transpose(1, 2)
                .to_concrete();

            let key: Tensor<4, F> = key.cast();
            if let Some(norm) = &self.attention_k_norm {
                norm.forward_generic_4d(&key)
            } else {
                key
            }
        };
        let value_states: Tensor<4, F> = {
            let attention_wv = self
                .attention_wv
                .as_ref()
                .expect("separate attention without V weights must use a shared KV cache");
            let mut value_states = hidden_f32.q_mat_mul(attention_wv);

            if let Some(bias) = &self.bias {
                let bias_f32: Tensor<1, f32> = bias.bias_v.cast();
                value_states = value_states.add_(&bias_f32);
            }

            let value_states: Tensor<4, F> = value_states
                .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
                .transpose(1, 2)
                .to_concrete()
                .cast();
            if let Some(norm) = &self.attention_v_norm {
                norm.forward_generic_4d(&value_states)
            } else {
                value_states
            }
        };

        let (query_states, key_states) = rope_cache.forward(
            &query_states,
            &key_states,
            start_pos,
            pos_ids,
            self.interleaved_rope,
        );
        (query_states, key_states, value_states)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_query<B>(
        &self,
        num_heads: usize,
        head_dim: usize,
        hidden_states: &Tensor<3, F, B>,
        rope_cache: &RopeImplementation<F>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> Tensor<4, F>
    where
        B: Fusion<3, F>,
    {
        let [b_sz, seq_len, _] = hidden_states.shape();
        let hidden_f32 = hidden_states.cast::<f32>();

        let query_states: Tensor<4, F> = if let Some(attention_qkv) = &self.attention_qkv {
            // Shared-KV callers only need Q, but a fused QKV weight forces us to
            // project K/V as well and discard them. No current shared-KV model
            // (Gemma 4 or the MTP assistant) uses a fused QKV weight, so this
            // branch is effectively unreachable today; if a future one does, it
            // pays a ~3x projection here and should grow a Q-only weight slice.
            let query_width = num_heads * head_dim;
            let mut qkv = hidden_f32.q_mat_mul(attention_qkv);
            if let Some(bias) = &self.bias {
                let bias_f32: Tensor<1, f32> = bias.bias_qkv.cast();
                qkv = qkv.add_(&bias_f32);
            }
            let query_states = qkv.narrow(D::Minus1, 0, query_width).to_concrete();
            let query = query_states
                .reshape([b_sz, seq_len, num_heads, head_dim])
                .transpose(1, 2)
                .to_concrete();
            let query: Tensor<4, F> = query.cast();
            if let Some(norm) = &self.attention_q_norm {
                norm.forward_generic_4d(&query)
            } else {
                query
            }
        } else {
            let mut query_states = hidden_f32.q_mat_mul(&self.attention_wq);
            if let Some(bias) = &self.bias {
                let bias_f32: Tensor<1, f32> = bias.bias_q.cast();
                query_states = query_states.add_(&bias_f32);
            }
            let query = query_states
                .reshape([b_sz, seq_len, num_heads, head_dim])
                .transpose(1, 2)
                .to_concrete();
            let query: Tensor<4, F> = query.cast();
            if let Some(norm) = &self.attention_q_norm {
                norm.forward_generic_4d(&query)
            } else {
                query
            }
        };

        let (query_states, _) = rope_cache.forward(
            &query_states,
            &query_states,
            start_pos,
            pos_ids,
            self.interleaved_rope,
        );
        query_states
    }
}

pub struct GroupedAttention {
    pub attention_qkv: QMatrix,
    pub interleaved_rope: bool,
}

impl GroupedAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward<F, B>(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        x: &Tensor<3, F, B>,
        rope_cache: &RopeImplementation<F>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> (Tensor<4, F>, Tensor<4, F>, Tensor<4, F>)
    where
        F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
        f32: CastTo<F> + CastTensor<F>,
        B: Fusion<3, F>,
    {
        let [b_sz, seq_len, _] = x.shape();
        // Compute in f32 for SIMD ops compatibility
        let x_f32 = x.cast::<f32>();
        let qkv = x_f32.q_mat_mul(&self.attention_qkv);

        let query_pos = num_heads * head_dim;
        let query_states = qkv.narrow(D::Minus1, 0, query_pos);
        let key_states = qkv.narrow(D::Minus1, query_pos, num_key_value_heads * head_dim);
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + num_key_value_heads * head_dim,
            num_key_value_heads * head_dim,
        );

        let query_states: Tensor<4, F> = query_states
            .reshape([b_sz, seq_len, num_heads, head_dim])
            .transpose(1, 2)
            .to_concrete()
            .cast();
        let key_states: Tensor<4, F> = key_states
            .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2)
            .to_concrete()
            .cast();
        let value_states: Tensor<4, F> = value_states
            .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2)
            .to_concrete()
            .cast();

        let (query_states, key_states) = rope_cache.forward(
            &query_states,
            &key_states,
            start_pos,
            pos_ids,
            self.interleaved_rope,
        );

        (query_states, key_states, value_states)
    }
}

pub struct LlamaAttention<F: FloatDataType + SimdElement = f32> {
    pub attention_variant: AttentionVariant<F>,
    pub attention_wo: Linear<F>,
    pub attention_norm: RmsNorm<1, F>,
    pub post_attention_norm: Option<RmsNorm<1, F>>,
    pub feed_forward_variant: FeedForwardVariant<F>,
    pub ffn_norm: RmsNorm<1, F>,
    pub post_ffn_norm: Option<RmsNorm<1, F>>,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub rope_cache: RopeImplementation<F>,
    pub(crate) sliding_window_size: Option<usize>,
    pub(crate) attention_scale: f32,
    pub(crate) shared_kv_layer: Option<usize>,
    pub(crate) per_layer_inp_gate: Option<QMatrix>,
    pub(crate) per_layer_proj: Option<QMatrix>,
    pub(crate) per_layer_post_norm: Option<RmsNorm<1, F>>,
    pub(crate) layer_output_scale: Option<Tensor<1, F>>,
}

impl<F: FloatDataType + SimdElement + Default> LlamaAttention<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    fn logical_kv_len(&self, start_pos: usize, q_len: usize) -> usize {
        let len = start_pos + q_len;
        self.sliding_window_size
            .map(|window| len.min(window))
            .unwrap_or(len)
    }

    fn narrow_kv_to_logical_len(
        &self,
        key_states: Tensor<4, f32>,
        value_states: Tensor<4, f32>,
        logical_kv_len: usize,
    ) -> (Tensor<4, f32>, Tensor<4, f32>) {
        if key_states.shape()[2] <= logical_kv_len {
            return (key_states, value_states);
        }

        (
            key_states.narrow(2, 0, logical_kv_len).to_concrete(),
            value_states.narrow(2, 0, logical_kv_len).to_concrete(),
        )
    }

    pub(crate) fn forward<B>(
        &self,
        hidden_states: &Tensor<3, F, B>,
        attention_mask: Option<&AttentionMask<f32>>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
        cache: Option<&mut KvCache<f32>>,
    ) -> Tensor<3, F>
    where
        B: Fusion<3, F>,
    {
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
                pos_ids,
            ),
            AttentionVariant::Grouped(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
                pos_ids,
            ),
        };

        // Convert to f32 for cache operations (cache uses f32 for SIMD compatibility)
        let query_f32: Tensor<4, f32> = query_states.cast();
        let key_f32: Tensor<4, f32> = key_states.cast();
        let value_f32: Tensor<4, f32> = value_states.cast();

        let (key_f32, value_f32) = match cache {
            None => (key_f32, value_f32),
            Some(cache) => cache.append(&query_f32.device(), &key_f32, &value_f32),
        };
        let (key_f32, value_f32) = self.narrow_kv_to_logical_len(
            key_f32,
            value_f32,
            self.logical_kv_len(start_pos, q_len),
        );

        forward_attention_qkv_f32(
            &query_f32,
            &key_f32,
            &value_f32,
            &self.attention_wo,
            attention_mask,
            b_sz,
            q_len,
            hidden_size,
            self.attention_scale,
        )
    }

    pub(crate) fn forward_with_shared_kv<B>(
        &self,
        hidden_states: &Tensor<3, F, B>,
        attention_mask: Option<&AttentionMask<f32>>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
        key_states: &Tensor<4, f32>,
        value_states: &Tensor<4, f32>,
    ) -> Tensor<3, F>
    where
        B: Fusion<3, F>,
    {
        let [b_sz, q_len, _] = hidden_states.shape();
        let query_states = match self.attention_variant {
            AttentionVariant::Separate(ref attention) => attention.forward_query(
                self.n_head,
                self.head_dim,
                hidden_states,
                &self.rope_cache,
                start_pos,
                pos_ids,
            ),
            AttentionVariant::Grouped(_) => {
                panic!("grouped attention cannot reuse a shared KV cache")
            }
        };
        let query_f32: Tensor<4, f32> = query_states.cast();
        let (key_states, value_states) = self.narrow_kv_to_logical_len(
            key_states.clone(),
            value_states.clone(),
            self.logical_kv_len(start_pos, q_len),
        );

        forward_attention_qkv_f32(
            &query_f32,
            &key_states,
            &value_states,
            &self.attention_wo,
            attention_mask,
            b_sz,
            q_len,
            self.hidden_size,
            self.attention_scale,
        )
    }

    #[cfg(feature = "vision")]
    pub(crate) fn forward_with_trace<B>(
        &self,
        hidden_states: &Tensor<3, F, B>,
        attention_mask: Option<&AttentionMask<f32>>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
        cache: Option<&mut KvCache<f32>>,
        layer_idx: usize,
    ) -> Tensor<3, F>
    where
        B: Fusion<3, F>,
    {
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
                pos_ids,
            ),
            AttentionVariant::Grouped(ref attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                &self.rope_cache,
                start_pos,
                pos_ids,
            ),
        };

        let query_f32: Tensor<4, f32> = query_states.cast();
        let key_f32: Tensor<4, f32> = key_states.cast();
        let value_f32: Tensor<4, f32> = value_states.cast();

        crate::raw::debug_check_nan_f32(&query_f32, layer_idx, "Q_pre_cache", start_pos);
        crate::raw::debug_check_nan_f32(&key_f32, layer_idx, "K_new", start_pos);
        crate::raw::debug_check_nan_f32(&value_f32, layer_idx, "V_new", start_pos);

        let (key_f32, value_f32) = match cache {
            None => (key_f32, value_f32),
            Some(cache) => cache.append(&query_f32.device(), &key_f32, &value_f32),
        };
        let (key_f32, value_f32) = self.narrow_kv_to_logical_len(
            key_f32,
            value_f32,
            self.logical_kv_len(start_pos, q_len),
        );

        crate::raw::debug_check_nan_f32(&key_f32, layer_idx, "K_cache_view", start_pos);
        crate::raw::debug_check_nan_f32(&value_f32, layer_idx, "V_cache_view", start_pos);

        let scale = self.attention_scale;
        let padded_attention_mask =
            attention_mask.and_then(|m| pad_attention_mask_to_kv_len(m, key_f32.shape()[2]));
        let attention_mask = padded_attention_mask.as_ref().or(attention_mask);
        let attn_raw = query_f32.flash_attention(
            &key_f32,
            &value_f32,
            scale,
            attention_mask.map(|m| {
                let kind = if m.is_strict_causal() {
                    fusor::MaskKind::Causal
                } else {
                    fusor::MaskKind::QKMask
                };
                (m.mask(), kind)
            }),
        );
        crate::raw::debug_check_nan_f32(&attn_raw, layer_idx, "flash_out", start_pos);
        let attn_output = attn_raw.transpose(1, 2);
        let attn_output = attn_output.reshape([b_sz, q_len, hidden_size]);
        let attn_output_f: Tensor<3, F> = attn_output.cast();
        let probe_in: fusor::Tensor<3, f32> = attn_output_f.clone().cast();
        crate::raw::debug_check_nan_f32(&probe_in, layer_idx, "before_wo", start_pos);
        let out = self.attention_wo.forward_generic(&attn_output_f);
        out
    }
}

fn pad_attention_mask_to_kv_len(
    attention_mask: &AttentionMask<f32>,
    kv_seq_len: usize,
) -> Option<AttentionMask<f32>> {
    let mask = attention_mask.mask();
    let [rows, cols] = mask.shape();
    if cols == kv_seq_len {
        return None;
    }

    if cols > kv_seq_len {
        let start_col = cols - kv_seq_len;
        return Some(AttentionMask::new(
            mask.narrow(1, start_col, kv_seq_len).to_concrete(),
        ));
    }

    let padded = Tensor::full(&mask.device(), [rows, kv_seq_len], f32::NEG_INFINITY);
    Some(AttentionMask::new(
        padded.slice_assign([0..rows, 0..cols], mask),
    ))
}

/// Forward attention QKV computation in f32 for SIMD compatibility.
/// All intermediate computation happens in f32, with the final result cast back to F.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_attention_qkv_f32<F>(
    query_states: &Tensor<4, f32>,
    key_states: &Tensor<4, f32>,
    value_states: &Tensor<4, f32>,
    attention_wo: &Linear<F>,
    attention_mask: Option<&AttentionMask<f32>>,
    b_sz: usize,
    q_len: usize,
    hidden_size: usize,
    attention_scale: f32,
) -> Tensor<3, F>
where
    F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    let padded_attention_mask =
        attention_mask.and_then(|m| pad_attention_mask_to_kv_len(m, key_states.shape()[2]));
    let attention_mask = padded_attention_mask.as_ref().or(attention_mask);
    let attn_output = query_states.flash_attention(
        key_states,
        value_states,
        attention_scale,
        attention_mask.map(|m| {
            let kind = if m.is_strict_causal() {
                fusor::MaskKind::Causal
            } else {
                fusor::MaskKind::QKMask
            };
            (m.mask(), kind)
        }),
    );

    let attn_output = attn_output.transpose(1, 2);

    let attn_output = attn_output.reshape([b_sz, q_len, hidden_size]);

    attention_wo.forward_generic(&attn_output.cast())
}
