//! ModernBERT transformer layer: a shared [`fusor::TransformerBlock`] (fused
//! QKV + RoPE + pre-norm LayerNorm + GeGLU) plus the sliding-window local
//! attention that ModernBERT alternates with global attention.

use fusor::layers::{LayerNorm, Linear};
use fusor::{
    AttentionVariant, Device, FeedForwardVariant, GatedActivation, GroupedAttention,
    LlamaFeedForward, Norm, Result, RopeCache, Tensor, TransformerBlock, VarBuilder,
};

use super::super::utils::MASK_NEG_VALUE;

/// Build an additive sliding-window bias `[seq, seq]`: `0` where the relative
/// distance `|i - j| <= window`, and a large negative value elsewhere so those
/// positions vanish after softmax. Shared across batch and heads.
fn sliding_window_bias(seq_len: usize, window: usize, device: &Device) -> Tensor<2, f32> {
    let mut data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if i.abs_diff(j) > window {
                data[i * seq_len + j] = MASK_NEG_VALUE;
            }
        }
    }
    Tensor::new(device, &data)
        .reshape([seq_len, seq_len])
        .to_concrete()
}

/// A single ModernBERT transformer layer.
///
/// Global layers run the shared block directly. Local layers reuse the block's
/// projections + RoPE but compute a windowed attention (a query at position `i`
/// attends only to keys within `window` positions), which the shared
/// BatchKey-masked path cannot express.
pub struct ModernBertLayer {
    block: TransformerBlock<f32, RopeCache>,
    /// Half-window for local layers; `None` selects global attention.
    window: Option<usize>,
    device: Device,
}

impl ModernBertLayer {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::config::ModernBertConfig,
        layer_idx: usize,
        rope_cache: &RopeCache,
        window: Option<usize>,
    ) -> Result<Self> {
        let eps = config.norm_eps;

        // Layer 0 has no attn_norm - it uses the embedding norm instead.
        let attention_norm = if layer_idx > 0 {
            Some(Norm::Layer(LayerNorm::load(
                device,
                &mut vb.pp("attn_norm"),
                eps,
            )?))
        } else {
            None
        };

        let wqkv = vb.get("attn_qkv.weight", device)?;
        let wo = vb.get("attn_output.weight", device)?;
        let ffn_norm = LayerNorm::load(device, &mut vb.pp("ffn_norm"), eps)?;

        let gate_up = vb.get("ffn_gate_up.weight", device)?;
        let down = vb.get("ffn_down.weight", device)?;

        let block = TransformerBlock {
            attention_variant: AttentionVariant::Grouped(GroupedAttention {
                attention_qkv: wqkv,
                interleaved_rope: false,
            }),
            attention_wo: Linear::new(wo, None),
            attention_norm,
            post_attention_norm: None,
            // ModernBERT uses a fused gate+up weight and GELU (GeGLU).
            feed_forward_variant: FeedForwardVariant::Llama(Box::new(
                LlamaFeedForward::from_fused_gated(gate_up, down, GatedActivation::GeLU),
            )),
            ffn_norm: Norm::Layer(ffn_norm),
            post_ffn_norm: None,
            n_head: config.num_heads,
            n_kv_head: config.num_kv_heads,
            head_dim: config.head_dimension,
            hidden_size: config.num_heads * config.head_dimension,
            rope_cache: rope_cache.clone(),
            sliding_window_size: None,
        };

        Ok(Self {
            block,
            window,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        mask_bias: Option<&Tensor<2, f32>>,
    ) -> Tensor<3, f32> {
        let [_, seq_len, _] = hidden_states.shape();
        match self.window {
            // Local layer whose window actually constrains the sequence.
            Some(window) if seq_len > window + 1 => {
                self.forward_windowed(hidden_states, window, mask_bias)
            }
            // Global attention, or a window wider than the sequence: the shared
            // BatchKey-masked block.
            _ => self.block.forward_block(hidden_states, mask_bias),
        }
    }

    /// Local sliding-window attention. Reuses the shared block's pre-norm,
    /// QKV+RoPE projection, output projection, and FFN, supplying its own
    /// per-batch banded attention in between.
    fn forward_windowed(
        &self,
        hidden_states: &Tensor<3, f32>,
        window: usize,
        pad_bias: Option<&Tensor<2, f32>>,
    ) -> Tensor<3, f32> {
        let block = &self.block;

        // Pre-norm (layer 0 input is already normed by the embedding norm).
        let normed = match &block.attention_norm {
            Some(norm) => norm.forward(hidden_states),
            None => hidden_states.clone(),
        };

        // Shared projection + RoPE.
        let (query_states, key_states, value_states) = block.attention_variant.forward(
            block.n_head,
            block.head_dim,
            block.n_kv_head,
            &normed,
            &block.rope_cache,
            0,
            None,
        );

        let [batch_size, _, seq_len, _] = query_states.shape();
        let scale = 1.0 / (block.head_dim as f32).sqrt();

        // The band mask is a shared `[q, k]` tensor, but per-sample padding lives
        // on the key axis, so the combined mask is logically `[batch, q, k]`. The
        // fused flash-attention kernel only accepts a 2D mask, so we fold the band
        // and each sample's padding into a per-element `QKMask` and run the batch
        // as a short loop (batch is typically 1 for single-text inference).
        let band = sliding_window_bias(seq_len, window, &self.device);
        let mut per_batch = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let q_b = query_states.narrow(0, b, 1).to_concrete();
            let k_b = key_states.narrow(0, b, 1).to_concrete();
            let v_b = value_states.narrow(0, b, 1).to_concrete();
            let mask_b = match pad_bias {
                // combined[i, j] = band[i, j] + padding[b, j]
                Some(pb) => {
                    let row = pb
                        .narrow(0, b, 1)
                        .broadcast_as([seq_len, seq_len])
                        .to_concrete();
                    (&band + &row).to_concrete()
                }
                None => band.clone(),
            };
            per_batch.push(q_b.flash_attention(
                &k_b,
                &v_b,
                scale,
                Some((&mask_b, fusor::MaskKind::QKMask)),
            ));
        }
        let attn_output = Tensor::cat(per_batch, 0);

        // Merge heads and project output.
        let attn_output = attn_output
            .transpose(1, 2)
            .to_concrete()
            .reshape([batch_size, seq_len, block.hidden_size])
            .to_concrete();
        let attn_output = block.attention_wo.forward(&attn_output);

        // Residual + pre-norm FFN + residual.
        let hidden = hidden_states.add_(&attn_output);
        let ffn_input = block.ffn_norm.forward(&hidden);
        let ffn_output = block.feed_forward_variant.forward(&ffn_input);
        hidden.add_(&ffn_output)
    }
}
