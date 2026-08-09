//! The Llama/Qwen decoder layer is the shared [`fusor::TransformerBlock`]
//! specialized to this crate's [`RopeImplementation`]. This module wires the
//! decoder's RoPE cache into the block via [`fusor::RopeLike`], aliases the
//! block as [`LlamaAttention`], and (under the `vision` feature) provides a
//! tracing wrapper that interleaves NaN probes through the attention sublayer.

use crate::raw::rope::RopeImplementation;
use fusor::{CastTensor, CastTo, FloatDataType, RopeLike, SimdElement, Tensor};

// Re-export the shared block building blocks so the rest of the crate keeps
// importing them from `attention_layer`.
pub(crate) use fusor::{
    AttentionBias, AttentionVariant, FeedForwardVariant, GroupedAttention, LlamaFeedForward, Norm,
    PhiFeedForward, SeparateAttention, TransformerBlock,
};

/// The decoder layer: the shared transformer block parameterized by this
/// crate's RoPE implementation (plain Llama RoPE or Qwen-VL multi-axis RoPE).
pub(crate) type LlamaAttention<F> = TransformerBlock<F, RopeImplementation<F>>;

impl<F> RopeLike<F> for RopeImplementation<F>
where
    F: FloatDataType + SimdElement + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    fn apply(
        &self,
        query: &Tensor<4, F>,
        key: &Tensor<4, F>,
        start_pos: usize,
        position_ids: Option<&Tensor<2, F>>,
        interleaved: bool,
    ) -> (Tensor<4, F>, Tensor<4, F>) {
        self.forward(query, key, start_pos, position_ids, interleaved)
    }
}

/// Attention sublayer with NaN probes interleaved between each step. A free
/// function (not a block method) because it calls this crate's debug helpers,
/// which cannot live in `fusor`. Mirrors [`TransformerBlock::forward`].
#[cfg(feature = "vision")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_with_trace<F, R, B>(
    block: &TransformerBlock<F, R>,
    hidden_states: &Tensor<3, F, B>,
    attention_mask: Option<&fusor::cache::AttentionMask<f32>>,
    start_pos: usize,
    pos_ids: Option<&Tensor<2, F>>,
    cache: Option<&mut fusor::cache::KvCache<f32>>,
    layer_idx: usize,
) -> Tensor<3, F>
where
    F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
    R: RopeLike<F>,
    B: fusor::Fusion<3, F>,
{
    let [b_sz, q_len, _] = hidden_states.shape();
    let hidden_size = block.hidden_size;

    let (query_states, key_states, value_states) = block.attention_variant.forward(
        block.n_head,
        block.head_dim,
        block.n_kv_head,
        hidden_states,
        &block.rope_cache,
        start_pos,
        pos_ids,
    );

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

    crate::raw::debug_check_nan_f32(&key_f32, layer_idx, "K_cache_view", start_pos);
    crate::raw::debug_check_nan_f32(&value_f32, layer_idx, "V_cache_view", start_pos);

    let scale = 1. / (block.head_dim as f64).sqrt();
    let attn_raw = query_f32.flash_attention(
        &key_f32,
        &value_f32,
        scale as f32,
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
    block.attention_wo.forward_generic(&attn_output_f)
}
