use crate::raw::rope::RopeImplementation;

use fusor::cache::AttentionMask;
use fusor::cache::KvCache;
use fusor::layers::Linear;
use fusor::layers::RmsNorm;
use fusor::CastTensor;
use fusor::CastTo;
use fusor::FloatDataType;
use fusor::QMatrix;
use fusor::SimdElement;
use fusor::Tensor;
use fusor::TensorBacking;
use fusor::D;

/// Helper function to do quantized matmul with a generic type F.
/// Converts F -> f32, performs q_mat_mul, then converts f32 -> F.
#[allow(dead_code)]
fn q_mat_mul_generic<const R: usize, F, B>(
    input: &Tensor<R, F, B>,
    weight: &QMatrix,
) -> Tensor<R, F>
where
    F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
    B: TensorBacking<R, Elem = F>,
{
    let input_f32 = input.cast::<f32>();
    let output_f32 = input_f32.q_mat_mul(weight);
    output_f32.cast()
}

pub enum FeedForwardVariant<F: FloatDataType + SimdElement = f32> {
    // Used by the Llama, Qwen, and Gemma models
    Llama(Box<LlamaFeedForward<F>>),
    // Used by the Phi models
    Phi(PhiFeedForward),
}

impl<F: FloatDataType + SimdElement + Default> FeedForwardVariant<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    pub(crate) fn forward<B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
    where
        B: TensorBacking<3, Elem = F>,
    {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward(x),
            FeedForwardVariant::Phi(ffn) => ffn.forward(x),
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
        B: TensorBacking<3, Elem = F>,
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
    gate: Option<QMatrix>,
    gate_len: usize,
    gate_up: Option<QMatrix>,
    gate_bias: Option<Tensor<1, F>>,
    down: QMatrix,
    down_bias: Option<Tensor<1, F>>,
    up: Option<QMatrix>,
    up_len: usize,
    up_bias: Option<Tensor<1, F>>,
}

impl<F: FloatDataType + SimdElement> LlamaFeedForward<F> {
    pub(crate) fn new(gate: QMatrix, down: QMatrix, up: QMatrix) -> Self {
        let gate_len = gate.shape()[0];
        let up_len = up.shape()[0];
        let gate_up = QMatrix::concat_rows(&[&gate, &up]);
        let (gate, up) = if gate_up.is_some() {
            (None, None)
        } else {
            (Some(gate), Some(up))
        };
        Self {
            gate,
            gate_len,
            gate_up,
            down,
            up,
            up_len,
            gate_bias: None,
            down_bias: None,
            up_bias: None,
        }
    }

    pub(crate) fn new_with_bias(
        gate: QMatrix,
        gate_bias: Option<Tensor<1, F>>,
        down: QMatrix,
        down_bias: Option<Tensor<1, F>>,
        up: QMatrix,
        up_bias: Option<Tensor<1, F>>,
    ) -> Self {
        let gate_len = gate.shape()[0];
        let up_len = up.shape()[0];
        let gate_up = QMatrix::concat_rows(&[&gate, &up]);
        let (gate, up) = if gate_up.is_some() {
            (None, None)
        } else {
            (Some(gate), Some(up))
        };
        Self {
            gate,
            gate_len,
            gate_up,
            gate_bias,
            down,
            down_bias,
            up,
            up_len,
            up_bias,
        }
    }

    pub(crate) fn forward<B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
    where
        F: CastTo<f32> + CastTensor<f32>,
        f32: CastTo<F> + CastTensor<F>,
        B: TensorBacking<3, Elem = F>,
    {
        // All computation happens in f32 for compatibility with SIMD ops
        let x_f32 = x.cast::<f32>();

        let [_b_sz, _seq_len, _hidden] = x.shape();
        let up_result = if let Some(gate_up) = &self.gate_up {
            // Natural unfused source. The compute-graph fuser auto-detects
            // the paired `silu(gate) * up` pattern (with optional bias
            // broadcasts) and rewrites this to a single paired-mode QMatMul kernel
            // that applies the epilogue in-register at epilogue time.
            let gate_up_states = x_f32.q_mat_mul(gate_up);
            let mut w1 = gate_up_states
                .narrow(D::Minus1, 0, self.gate_len)
                .to_concrete();
            let mut w3 = gate_up_states
                .narrow(D::Minus1, self.gate_len, self.up_len)
                .to_concrete();
            if let Some(ref bias) = self.gate_bias {
                let bias_f32: Tensor<1, f32> = bias.cast();
                w1 = w1.add_(&bias_f32);
            }
            if let Some(ref bias) = self.up_bias {
                let bias_f32: Tensor<1, f32> = bias.cast();
                w3 = w3.add_(&bias_f32);
            }
            (w1.silu() * w3).to_concrete()
        } else {
            let gate = self
                .gate
                .as_ref()
                .expect("separate gate matrix should exist without fused gate_up");
            let mut w1 = x_f32.q_mat_mul(gate);
            if let Some(ref bias) = self.gate_bias {
                let bias_f32: Tensor<1, f32> = bias.cast();
                w1 = w1.add_(&bias_f32);
            }

            let up = self
                .up
                .as_ref()
                .expect("separate up matrix should exist without fused gate_up");
            let mut w3 = x_f32.q_mat_mul(up);
            if let Some(ref bias) = self.up_bias {
                let bias_f32: Tensor<1, f32> = bias.cast();
                w3 = w3.add_(&bias_f32);
            }

            (w1.silu() * w3).to_concrete()
        };
        let mut up = up_result.q_mat_mul(&self.down);
        if let Some(ref bias) = self.down_bias {
            let bias_f32: Tensor<1, f32> = bias.cast();
            up = up.add_(&bias_f32);
        }

        // Cast back to F
        up.cast()
    }
}

pub enum AttentionVariant<F: FloatDataType + SimdElement = f32> {
    Separate(Box<SeparateAttention<F>>),
    Grouped(GroupedAttention<F>),
    Paired(PairedAttention<F>),
}

pub struct AttentionBias<F: FloatDataType + SimdElement = f32> {
    bias_q: Tensor<1, F>,
    bias_k: Tensor<1, F>,
    bias_v: Tensor<1, F>,
}

impl<F: FloatDataType + SimdElement> AttentionBias<F> {
    pub fn new(q: Tensor<1, F>, k: Tensor<1, F>, v: Tensor<1, F>) -> Self {
        Self {
            bias_q: q,
            bias_k: k,
            bias_v: v,
        }
    }
}

pub struct SeparateAttention<F: FloatDataType + SimdElement = f32> {
    pub attention_wq: QMatrix,
    pub attention_q_norm: Option<RmsNorm<1, F>>,
    pub attention_wk: QMatrix,
    pub attention_k_norm: Option<RmsNorm<1, F>>,
    pub attention_wv: QMatrix,
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
        B: TensorBacking<3, Elem = F>,
    {
        let [b_sz, seq_len, _] = hidden_states.shape();

        // Compute in f32 for SIMD ops compatibility
        let hidden_f32 = hidden_states.cast::<f32>();

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
            let mut key_states = hidden_f32.q_mat_mul(&self.attention_wk);

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
            let mut value_states = hidden_f32.q_mat_mul(&self.attention_wv);

            if let Some(bias) = &self.bias {
                let bias_f32: Tensor<1, f32> = bias.bias_v.cast();
                value_states = value_states.add_(&bias_f32);
            }

            value_states
                .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
                .transpose(1, 2)
                .to_concrete()
                .cast()
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
}

pub struct GroupedAttention<F: FloatDataType + SimdElement = f32> {
    pub attention_qkv: QMatrix,
    pub attention_q_norm: Option<RmsNorm<1, F>>,
    pub attention_k_norm: Option<RmsNorm<1, F>>,
    pub bias: Option<AttentionBias<F>>,
    pub interleaved_rope: bool,
}

impl<F: FloatDataType + SimdElement + Default> GroupedAttention<F>
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
        x: &Tensor<3, F, B>,
        rope_cache: &RopeImplementation<F>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> (Tensor<4, F>, Tensor<4, F>, Tensor<4, F>)
    where
        B: TensorBacking<3, Elem = F>,
    {
        let [b_sz, seq_len, _] = x.shape();
        // Compute in f32 for SIMD ops compatibility
        let x_f32 = x.cast::<f32>();
        let qkv = x_f32.q_mat_mul(&self.attention_qkv);

        let query_pos = num_heads * head_dim;
        let kv_pos = num_key_value_heads * head_dim;
        let mut query_states = qkv.narrow(D::Minus1, 0, query_pos).to_concrete();
        let mut key_states = qkv.narrow(D::Minus1, query_pos, kv_pos).to_concrete();
        let mut value_states = qkv
            .narrow(D::Minus1, query_pos + kv_pos, kv_pos)
            .to_concrete();

        if let Some(bias) = &self.bias {
            let bq: Tensor<1, f32> = bias.bias_q.cast();
            let bk: Tensor<1, f32> = bias.bias_k.cast();
            let bv: Tensor<1, f32> = bias.bias_v.cast();
            query_states = query_states.add_(&bq);
            key_states = key_states.add_(&bk);
            value_states = value_states.add_(&bv);
        }

        let mut query_states: Tensor<4, F> = query_states
            .reshape([b_sz, seq_len, num_heads, head_dim])
            .transpose(1, 2)
            .to_concrete()
            .cast();
        let mut key_states: Tensor<4, F> = key_states
            .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2)
            .to_concrete()
            .cast();
        let value_states: Tensor<4, F> = value_states
            .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2)
            .to_concrete()
            .cast();

        if let Some(norm) = &self.attention_q_norm {
            query_states = norm.forward_generic_4d(&query_states);
        }
        if let Some(norm) = &self.attention_k_norm {
            key_states = norm.forward_generic_4d(&key_states);
        }

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

#[derive(Clone, Copy)]
pub enum PairedAttentionKind {
    QueryKey,
    QueryValue,
    KeyValue,
}

pub struct PairedAttention<F: FloatDataType + SimdElement = f32> {
    pub attention_pair: QMatrix,
    pub attention_single: QMatrix,
    pub attention_q_norm: Option<RmsNorm<1, F>>,
    pub attention_k_norm: Option<RmsNorm<1, F>>,
    pub bias: Option<AttentionBias<F>>,
    pub kind: PairedAttentionKind,
    pub interleaved_rope: bool,
}

impl<F: FloatDataType + SimdElement + Default> PairedAttention<F>
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
        x: &Tensor<3, F, B>,
        rope_cache: &RopeImplementation<F>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> (Tensor<4, F>, Tensor<4, F>, Tensor<4, F>)
    where
        B: TensorBacking<3, Elem = F>,
    {
        let [b_sz, seq_len, _] = x.shape();
        let query_pos = num_heads * head_dim;
        let kv_pos = num_key_value_heads * head_dim;
        let x_f32 = x.cast::<f32>();
        let pair = x_f32.q_mat_mul(&self.attention_pair);
        let single = x_f32.q_mat_mul(&self.attention_single);

        let (mut query_states, mut key_states, mut value_states) = match self.kind {
            PairedAttentionKind::QueryKey => {
                let query = pair.narrow(D::Minus1, 0, query_pos);
                let key = pair.narrow(D::Minus1, query_pos, kv_pos);
                (query.to_concrete(), key.to_concrete(), single.to_concrete())
            }
            PairedAttentionKind::QueryValue => {
                let query = pair.narrow(D::Minus1, 0, query_pos);
                let value = pair.narrow(D::Minus1, query_pos, kv_pos);
                (
                    query.to_concrete(),
                    single.to_concrete(),
                    value.to_concrete(),
                )
            }
            PairedAttentionKind::KeyValue => {
                let key = pair.narrow(D::Minus1, 0, kv_pos);
                let value = pair.narrow(D::Minus1, kv_pos, kv_pos);
                (single.to_concrete(), key.to_concrete(), value.to_concrete())
            }
        };

        if let Some(bias) = &self.bias {
            let bq: Tensor<1, f32> = bias.bias_q.cast();
            let bk: Tensor<1, f32> = bias.bias_k.cast();
            let bv: Tensor<1, f32> = bias.bias_v.cast();
            query_states = query_states.add_(&bq);
            key_states = key_states.add_(&bk);
            value_states = value_states.add_(&bv);
        }

        let mut query_states: Tensor<4, F> = query_states
            .reshape([b_sz, seq_len, num_heads, head_dim])
            .transpose(1, 2)
            .to_concrete()
            .cast();
        let mut key_states: Tensor<4, F> = key_states
            .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2)
            .to_concrete()
            .cast();
        let value_states: Tensor<4, F> = value_states
            .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
            .transpose(1, 2)
            .to_concrete()
            .cast();

        if let Some(norm) = &self.attention_q_norm {
            query_states = norm.forward_generic_4d(&query_states);
        }
        if let Some(norm) = &self.attention_k_norm {
            key_states = norm.forward_generic_4d(&key_states);
        }

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
}

impl<F: FloatDataType + SimdElement + Default> LlamaAttention<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    pub(crate) fn forward<B>(
        &self,
        hidden_states: &Tensor<3, F, B>,
        attention_mask: Option<&AttentionMask<f32>>,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
        cache: Option<&mut KvCache<f32>>,
    ) -> Tensor<3, F>
    where
        B: TensorBacking<3, Elem = F>,
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
            AttentionVariant::Paired(ref attention) => attention.forward(
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

        forward_attention_qkv_f32(
            &query_f32,
            &key_f32,
            &value_f32,
            &self.attention_wo,
            attention_mask,
            head_dim,
            b_sz,
            q_len,
            hidden_size,
        )
    }
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
    head_dim: usize,
    b_sz: usize,
    q_len: usize,
    hidden_size: usize,
) -> Tensor<3, F>
where
    F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    let scale = 1. / (head_dim as f64).sqrt();
    let attn_output = query_states.flash_attention(
        key_states,
        value_states,
        scale as f32,
        attention_mask.map(|m| (m.mask(), fusor::MaskKind::QKMask)),
    );

    let attn_output = attn_output.transpose(1, 2);

    let attn_output = attn_output.reshape([b_sz, q_len, hidden_size]);

    attention_wo.forward_generic(&attn_output.cast())
}
