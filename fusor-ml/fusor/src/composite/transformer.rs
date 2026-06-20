//! Shared pre-norm transformer block.
//!
//! A single parameterized attention + feed-forward block used by every
//! transformer model in the workspace (the Llama/Qwen decoders, the Qwen
//! vision tower, and the ModernBERT / Qwen encoders). The block composes the
//! low-level fusor primitives (`flash_attention`, RoPE, `RmsNorm`/`LayerNorm`,
//! `KvCache`, quantized `Linear`) behind a small set of enums so the same code
//! serves causal decoders and bidirectional encoders alike:
//!
//! * [`AttentionVariant`] — fused (`Grouped`) vs separate (`Separate`) Q/K/V
//!   projection, with optional grouped-query attention and optional q/k norm.
//! * [`FeedForwardVariant`] — gated SwiGLU/GeGLU ([`LlamaFeedForward`]) vs the
//!   plain split-gate Phi MLP ([`PhiFeedForward`]); the gated activation is
//!   selectable via [`GatedActivation`].
//! * [`Norm`] — `RmsNorm` or `LayerNorm`, so decoders (RMS) and BERT-style
//!   encoders (Layer) share the same block.
//! * [`RopeLike`] — abstracts the rotary cache so the block is agnostic to the
//!   concrete RoPE implementation (plain [`RopeCache`] or a model-specific one).
//!
//! The decoder hot path uses [`TransformerBlock::forward`] (attention sublayer
//! only; the model loop orchestrates norms + residuals so it can fuse them).
//! Encoders use [`TransformerBlock::forward_block`], which runs the full
//! pre-norm block in one call.

use crate::cache::AttentionMask;
use crate::cache::KvCache;
use crate::layers::{LayerNorm, Linear, RmsNorm};
use crate::MaskKind;
use crate::RopeCache;
use crate::Tensor;
use crate::D;
use crate::QMatrix;
use crate::{CastTensor, CastTo, FloatDataType, Fusion, SimdElement};

/// Abstracts a rotary-position-embedding cache so [`TransformerBlock`] does not
/// depend on any one model's RoPE implementation. Implemented for the plain
/// [`RopeCache`]; model-specific caches (e.g. multi-axis vision RoPE) provide
/// their own impl.
pub trait RopeLike<F: FloatDataType + SimdElement> {
    /// Apply RoPE to the query and key tensors, returning the rotated pair.
    fn apply(
        &self,
        query: &Tensor<4, F>,
        key: &Tensor<4, F>,
        start_pos: usize,
        position_ids: Option<&Tensor<2, F>>,
        interleaved: bool,
    ) -> (Tensor<4, F>, Tensor<4, F>);
}

impl<F> RopeLike<F> for RopeCache
where
    F: FloatDataType + SimdElement + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    fn apply(
        &self,
        query: &Tensor<4, F>,
        key: &Tensor<4, F>,
        start_pos: usize,
        _position_ids: Option<&Tensor<2, F>>,
        interleaved: bool,
    ) -> (Tensor<4, F>, Tensor<4, F>) {
        let q_f32: Tensor<4, f32> = query.cast();
        let k_f32: Tensor<4, f32> = key.cast();
        let (q_out, k_out) = if interleaved {
            self.forward_interleaved(&q_f32, &k_f32, start_pos)
        } else {
            self.forward(&q_f32, &k_f32, start_pos)
        };
        (q_out.cast(), k_out.cast())
    }
}

/// Activation applied to the gate branch of a gated feed-forward network.
#[derive(Clone, Copy, Debug)]
pub enum GatedActivation {
    /// SiLU / Swish — used by Llama, Qwen, Gemma (SwiGLU).
    SiLU,
    /// GELU — used by ModernBERT (GeGLU).
    GeLU,
}

impl GatedActivation {
    #[inline]
    fn apply(self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        match self {
            GatedActivation::SiLU => x.silu(),
            GatedActivation::GeLU => x.gelu().to_concrete(),
        }
    }
}

/// Normalization layer used inside a [`TransformerBlock`]. `RmsNorm` for
/// decoders / Qwen encoders, `LayerNorm` (f32) for BERT-style encoders.
pub enum Norm<F: FloatDataType + SimdElement = f32> {
    /// Root-mean-square normalization (no centering).
    Rms(RmsNorm<1, F>),
    /// Standard layer normalization (computed in f32).
    Layer(LayerNorm<1, f32>),
}

impl<F: FloatDataType + SimdElement + Default> Norm<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    /// Pre-normalization of a hidden state.
    pub fn forward<B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
    where
        B: Fusion<3, F>,
    {
        match self {
            Norm::Rms(n) => n.forward_generic(x),
            Norm::Layer(n) => {
                let x_f32 = x.cast::<f32>();
                let out: Tensor<3, f32> = n.forward(&x_f32).to_concrete();
                out.cast()
            }
        }
    }

    /// Fused `(input + residual)` followed by normalization. The `Rms` arm uses
    /// the fused residual kernel (the decode hot path depends on this); the
    /// `Layer` arm falls back to an explicit add (no `LayerNorm` fused-residual
    /// kernel exists, and encoders never relied on one).
    pub fn forward_residual_f32<B1, B2>(
        &self,
        input: &Tensor<3, f32, B1>,
        residual: &Tensor<3, f32, B2>,
    ) -> Tensor<3, F>
    where
        B1: Fusion<3, f32>,
        B2: Fusion<3, f32>,
    {
        match self {
            Norm::Rms(n) => n.forward_residual_f32(input, residual),
            Norm::Layer(n) => {
                let sum = input.add_(residual);
                let out: Tensor<3, f32> = n.forward(&sum).to_concrete();
                out.cast()
            }
        }
    }
}

/// Gated vs plain feed-forward selection.
pub enum FeedForwardVariant<F: FloatDataType + SimdElement = f32> {
    /// Gated FFN (SwiGLU / GeGLU) — Llama, Qwen, Gemma, ModernBERT.
    Llama(Box<LlamaFeedForward<F>>),
    /// Plain split-gate FFN — Phi models.
    Phi(PhiFeedForward),
}

impl<F: FloatDataType + SimdElement + Default> FeedForwardVariant<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    /// Feed-forward forward pass.
    pub fn forward<B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
    where
        B: Fusion<3, F>,
    {
        match self {
            FeedForwardVariant::Llama(ffn) => ffn.forward(x),
            FeedForwardVariant::Phi(ffn) => ffn.forward(x),
        }
    }

    /// Feed-forward pass that folds two residual adds into the down-projection
    /// epilogue (decode fast path). Returns `None` when the fold does not apply.
    pub fn forward_add_residuals<B, B1, B2>(
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

/// Plain split-gate feed-forward used by Phi models.
pub struct PhiFeedForward {
    /// Up projection (its output is split into gate + up halves).
    pub up: QMatrix,
    /// Down projection.
    pub down: QMatrix,
    /// Intermediate width (half of the up projection output).
    pub feed_forward_length: usize,
}

impl PhiFeedForward {
    fn forward<F, B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
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

/// Gated feed-forward (SwiGLU / GeGLU). Supports a pre-fused `gate_up`
/// projection (single matmul) or separate gate/up projections, with optional
/// per-projection biases, and a selectable [`GatedActivation`].
pub struct LlamaFeedForward<F: FloatDataType + SimdElement = f32> {
    gate: Option<QMatrix>,
    gate_up: Option<QMatrix>,
    gate_bias: Option<Tensor<1, F>>,
    down: QMatrix,
    down_bias: Option<Tensor<1, F>>,
    up: Option<QMatrix>,
    up_bias: Option<Tensor<1, F>>,
    activation: GatedActivation,
}

impl<F: FloatDataType + SimdElement> LlamaFeedForward<F> {
    /// Gated FFN from separate gate/up/down projections (SiLU activation).
    pub fn new(gate: QMatrix, down: QMatrix, up: QMatrix) -> Self {
        let gate_up = QMatrix::concat_rows(&[&gate, &up]);
        Self {
            gate: Some(gate),
            gate_up,
            down,
            up: Some(up),
            gate_bias: None,
            down_bias: None,
            up_bias: None,
            activation: GatedActivation::SiLU,
        }
    }

    /// Gated FFN from a pre-fused `[2 * intermediate, hidden]` gate+up
    /// projection and a down projection, with a selectable activation. Used by
    /// ModernBERT (GeGLU) where the gate and up weights are stored fused.
    pub fn from_fused_gated(gate_up: QMatrix, down: QMatrix, activation: GatedActivation) -> Self {
        Self {
            gate: None,
            gate_up: Some(gate_up),
            down,
            up: None,
            gate_bias: None,
            down_bias: None,
            up_bias: None,
            activation,
        }
    }

    /// Gated FFN with optional per-projection biases (SiLU activation).
    pub fn new_with_bias(
        gate: QMatrix,
        gate_bias: Option<Tensor<1, F>>,
        down: QMatrix,
        down_bias: Option<Tensor<1, F>>,
        up: QMatrix,
        up_bias: Option<Tensor<1, F>>,
    ) -> Self {
        let gate_up = QMatrix::concat_rows(&[&gate, &up]);
        Self {
            gate: Some(gate),
            gate_up,
            gate_bias,
            down,
            down_bias,
            up: Some(up),
            up_bias,
            activation: GatedActivation::SiLU,
        }
    }

    /// Gated feed-forward pass: `down(act(gate(x)) * up(x))`.
    pub fn forward<B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
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

    fn forward_add_residuals<B, B1, B2>(
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
                // SwiGLU/GeGLU split/gate authored in natural graph form: the
                // resolver folds `act(gate) * up` over the two narrow halves into
                // the qmatmul accumulator-offset epilogue (one dispatch on decode).
                let pair_len = gate_up.shape()[0] / 2;
                let projected = x_f32.q_mat_mul(gate_up);
                let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
                let up = projected
                    .narrow(D::Minus1, pair_len, pair_len)
                    .to_concrete();
                (self.activation.apply(&gate) * up).to_concrete()
            }
            Some(gate_up) => {
                let gate_width = self.gate.as_ref().expect("gated ffn gate").shape()[0];
                let up_width = self.up.as_ref().expect("gated ffn up").shape()[0];
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

                (self.activation.apply(&gate_states) * up_states).to_concrete()
            }
            None => {
                let gate = self.gate.as_ref().expect("gated ffn gate");
                let up = self.up.as_ref().expect("gated ffn up");
                let mut w1 = x_f32.q_mat_mul(gate);
                if let Some(ref bias) = self.gate_bias {
                    let bias_f32: Tensor<1, f32> = bias.cast();
                    w1 = w1.add_(&bias_f32);
                }
                let w1 = self.activation.apply(&w1);

                let mut w3 = x_f32.q_mat_mul(up);
                if let Some(ref bias) = self.up_bias {
                    let bias_f32: Tensor<1, f32> = bias.cast();
                    w3 = w3.add_(&bias_f32);
                }

                (w1 * w3).to_concrete()
            }
        }
    }
}

/// Fused vs separate Q/K/V projection.
pub enum AttentionVariant<F: FloatDataType + SimdElement = f32> {
    /// Separate Q/K/V projections (optionally a fused weight + optional q/k norm).
    Separate(Box<SeparateAttention<F>>),
    /// Single fused Q/K/V projection.
    Grouped(GroupedAttention),
}

impl<F: FloatDataType + SimdElement + Default> AttentionVariant<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    /// Project + RoPE the hidden states into `(query, key, value)` head tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn forward<B, R>(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        hidden_states: &Tensor<3, F, B>,
        rope: &R,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> (Tensor<4, F>, Tensor<4, F>, Tensor<4, F>)
    where
        B: Fusion<3, F>,
        R: RopeLike<F>,
    {
        match self {
            AttentionVariant::Separate(attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                rope,
                start_pos,
                pos_ids,
            ),
            AttentionVariant::Grouped(attention) => attention.forward(
                num_heads,
                head_dim,
                num_key_value_heads,
                hidden_states,
                rope,
                start_pos,
                pos_ids,
            ),
        }
    }
}

/// Optional additive biases for the Q/K/V projections.
pub struct AttentionBias<F: FloatDataType + SimdElement = f32> {
    bias_q: Tensor<1, F>,
    bias_k: Tensor<1, F>,
    bias_v: Tensor<1, F>,
    bias_qkv: Tensor<1, F>,
}

impl<F: FloatDataType + SimdElement + Default> AttentionBias<F> {
    /// Build an attention bias from separate q/k/v biases (also concatenated
    /// for the fused-QKV path).
    pub fn new(q: Tensor<1, F>, k: Tensor<1, F>, v: Tensor<1, F>) -> Self {
        let bias_qkv = crate::cat([q.clone(), k.clone(), v.clone()], 0).to_concrete();
        Self {
            bias_q: q,
            bias_k: k,
            bias_v: v,
            bias_qkv,
        }
    }
}

/// Separate Q/K/V projection (optionally backed by a fused weight, optionally
/// with per-head q/k normalization, optionally with biases).
pub struct SeparateAttention<F: FloatDataType + SimdElement = f32> {
    /// Query projection.
    pub attention_wq: QMatrix,
    /// Optional fused Q/K/V projection (used when present).
    pub attention_qkv: Option<QMatrix>,
    /// Optional per-head query normalization.
    pub attention_q_norm: Option<RmsNorm<1, F>>,
    /// Key projection.
    pub attention_wk: QMatrix,
    /// Optional per-head key normalization.
    pub attention_k_norm: Option<RmsNorm<1, F>>,
    /// Value projection.
    pub attention_wv: QMatrix,
    /// Optional projection biases.
    pub bias: Option<AttentionBias<F>>,
    /// Whether RoPE pairs adjacent elements (interleaved) or halves.
    pub interleaved_rope: bool,
}

impl<F: FloatDataType + SimdElement + Default> SeparateAttention<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    #[allow(clippy::too_many_arguments)]
    fn forward<B, R>(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        hidden_states: &Tensor<3, F, B>,
        rope: &R,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> (Tensor<4, F>, Tensor<4, F>, Tensor<4, F>)
    where
        B: Fusion<3, F>,
        R: RopeLike<F>,
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

                value_states
                    .reshape([b_sz, seq_len, num_key_value_heads, head_dim])
                    .transpose(1, 2)
                    .to_concrete()
                    .cast()
            };

            let (query_states, key_states) =
                rope.apply(&query_states, &key_states, start_pos, pos_ids, self.interleaved_rope);
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

        let (query_states, key_states) =
            rope.apply(&query_states, &key_states, start_pos, pos_ids, self.interleaved_rope);
        (query_states, key_states, value_states)
    }
}

/// Single fused Q/K/V projection (grouped-query friendly).
pub struct GroupedAttention {
    /// Fused `[(q + k + v) heads * head_dim, hidden]` projection.
    pub attention_qkv: QMatrix,
    /// Whether RoPE pairs adjacent elements (interleaved) or halves.
    pub interleaved_rope: bool,
}

impl GroupedAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward<F, B, R>(
        &self,
        num_heads: usize,
        head_dim: usize,
        num_key_value_heads: usize,
        x: &Tensor<3, F, B>,
        rope: &R,
        start_pos: usize,
        pos_ids: Option<&Tensor<2, F>>,
    ) -> (Tensor<4, F>, Tensor<4, F>, Tensor<4, F>)
    where
        F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
        f32: CastTo<F> + CastTensor<F>,
        B: Fusion<3, F>,
        R: RopeLike<F>,
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

        let (query_states, key_states) =
            rope.apply(&query_states, &key_states, start_pos, pos_ids, self.interleaved_rope);

        (query_states, key_states, value_states)
    }
}

/// A complete pre-norm transformer block: attention sublayer + gated/plain FFN,
/// each with its own normalization and (optional) post-normalization.
///
/// Generic over the float type `F` (decoders may run f16; encoders are f32) and
/// the rotary cache `R` (so a vision tower can supply multi-axis RoPE).
pub struct TransformerBlock<F: FloatDataType + SimdElement = f32, R = RopeCache> {
    /// Q/K/V projection variant.
    pub attention_variant: AttentionVariant<F>,
    /// Output projection.
    pub attention_wo: Linear<F>,
    /// Pre-attention normalization (`None` only for blocks whose input is
    /// pre-normalized upstream, e.g. ModernBERT layer 0).
    pub attention_norm: Option<Norm<F>>,
    /// Optional post-attention normalization.
    pub post_attention_norm: Option<Norm<F>>,
    /// Feed-forward variant.
    pub feed_forward_variant: FeedForwardVariant<F>,
    /// Pre-FFN normalization.
    pub ffn_norm: Norm<F>,
    /// Optional post-FFN normalization.
    pub post_ffn_norm: Option<Norm<F>>,
    /// Number of query heads.
    pub n_head: usize,
    /// Number of key/value heads (== `n_head` for MHA, fewer for GQA).
    pub n_kv_head: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Model hidden size.
    pub hidden_size: usize,
    /// Rotary cache.
    pub rope_cache: R,
    /// Sliding-window size for local-attention decoder layers.
    pub sliding_window_size: Option<usize>,
}

impl<F: FloatDataType + SimdElement + Default, R> TransformerBlock<F, R>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
    R: RopeLike<F>,
{
    /// Attention sublayer only (Q/K/V projection + RoPE + KV cache + flash
    /// attention + output projection). The decoder model loop applies the
    /// surrounding norms and residuals so it can fuse them; the hidden state
    /// passed in is already pre-normalized.
    pub fn forward<B>(
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

        let (query_states, key_states, value_states) = self.attention_variant.forward(
            self.n_head,
            self.head_dim,
            self.n_kv_head,
            hidden_states,
            &self.rope_cache,
            start_pos,
            pos_ids,
        );

        // Convert to f32 for cache operations (cache uses f32 for SIMD compatibility)
        let query_f32: Tensor<4, f32> = query_states.cast();
        let key_f32: Tensor<4, f32> = key_states.cast();
        let value_f32: Tensor<4, f32> = value_states.cast();

        let (key_f32, value_f32) = match cache {
            None => (key_f32, value_f32),
            Some(cache) => cache.append(&query_f32.device(), &key_f32, &value_f32),
        };

        let mask = attention_mask.map(causal_mask_tuple);

        forward_attention_qkv_f32(
            &query_f32,
            &key_f32,
            &value_f32,
            &self.attention_wo,
            mask,
            self.head_dim,
            b_sz,
            q_len,
            hidden_size,
        )
    }

    /// Full pre-norm block in a single call (used by bidirectional encoders):
    /// `norm → attention → residual → norm → FFN → residual`, with a
    /// `[batch, key]` padding bias applied as a [`MaskKind::BatchKeyMask`]. No
    /// KV cache; positions start at 0.
    pub fn forward_block(
        &self,
        hidden_states: &Tensor<3, F>,
        mask_bias: Option<&Tensor<2, f32>>,
    ) -> Tensor<3, F>
    where
        crate::AddOp: crate::SimdBinaryOp<F>,
    {
        let attn = self.attention_sublayer(hidden_states, mask_bias);
        let hidden = hidden_states.add_(&attn);

        let ffn_input = self.ffn_norm.forward(&hidden);
        let ffn_output = self.feed_forward_variant.forward(&ffn_input);
        hidden.add_(&ffn_output)
    }

    /// Pre-norm attention sublayer (`norm → attention → output projection`),
    /// without the residual add. Exposed so encoders with bespoke masking
    /// (e.g. ModernBERT's sliding window) can reuse the shared projection +
    /// RoPE while supplying their own attention computation around it.
    pub fn attention_sublayer(
        &self,
        hidden_states: &Tensor<3, F>,
        mask_bias: Option<&Tensor<2, f32>>,
    ) -> Tensor<3, F> {
        let [b_sz, seq_len, _] = hidden_states.shape();
        let normed = match &self.attention_norm {
            Some(n) => n.forward(hidden_states),
            None => hidden_states.clone(),
        };
        let (query_states, key_states, value_states) = self.attention_variant.forward(
            self.n_head,
            self.head_dim,
            self.n_kv_head,
            &normed,
            &self.rope_cache,
            0,
            None,
        );
        let query_f32: Tensor<4, f32> = query_states.cast();
        let key_f32: Tensor<4, f32> = key_states.cast();
        let value_f32: Tensor<4, f32> = value_states.cast();
        let mask = mask_bias.map(|m| (m, MaskKind::BatchKeyMask));
        forward_attention_qkv_f32(
            &query_f32,
            &key_f32,
            &value_f32,
            &self.attention_wo,
            mask,
            self.head_dim,
            b_sz,
            seq_len,
            self.hidden_size,
        )
    }
}

/// Map a high-level [`AttentionMask`] to the `(mask, kind)` tuple consumed by
/// `flash_attention`: strictly-causal masks use the GPU-optimized causal kernel,
/// others fall back to an explicit Q×K additive mask.
fn causal_mask_tuple(m: &AttentionMask<f32>) -> (&Tensor<2, f32>, MaskKind) {
    let kind = if m.is_strict_causal() {
        MaskKind::Causal
    } else {
        MaskKind::QKMask
    };
    (m.mask(), kind)
}

/// Flash attention over `[batch, heads, seq, head_dim]` q/k/v followed by the
/// output projection. Computation is in f32 (SIMD compatibility); the result is
/// cast back to `F`.
#[allow(clippy::too_many_arguments)]
pub fn forward_attention_qkv_f32<F>(
    query_states: &Tensor<4, f32>,
    key_states: &Tensor<4, f32>,
    value_states: &Tensor<4, f32>,
    attention_wo: &Linear<F>,
    mask: Option<(&Tensor<2, f32>, MaskKind)>,
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
    let attn_output = query_states.flash_attention(key_states, value_states, scale as f32, mask);

    let attn_output = attn_output.transpose(1, 2);

    let attn_output = attn_output.reshape([b_sz, q_len, hidden_size]);

    attention_wo.forward_generic(&attn_output.cast())
}
