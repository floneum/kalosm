//! ModernBERT self-attention with RoPE and fused QKV.

use fusor::{Device, QMatrix, Result, RopeCache, Tensor, VarBuilder};

use super::super::utils::{attention_mask_to_bias, merge_heads, split_heads, MASK_NEG_VALUE};

/// ModernBERT self-attention with fused QKV projection and RoPE.
pub struct ModernBertAttention {
    /// Fused QKV projection: [3 * hidden_size, hidden_size]
    wqkv: QMatrix,
    wo: QMatrix,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device: Device,
}

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

impl ModernBertAttention {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        _eps: f32,
    ) -> Result<Self> {
        // Fused QKV weight
        let wqkv = vb.get("attn_qkv.weight", device)?;
        let wo = vb.get("attn_output.weight", device)?;

        Ok(Self {
            wqkv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            device: device.clone(),
        })
    }

    /// Self-attention.
    ///
    /// `window` selects local sliding-window attention (`Some(half_width)`) vs
    /// full global attention (`None`). For local layers a query at position `i`
    /// attends only to keys `j` with `|i - j| <= half_width`.
    pub fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        rope_cache: &RopeCache,
        window: Option<usize>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        let hidden_size = self.num_heads * self.head_dim;

        // Compute fused QKV projection: [batch, seq_len, 3 * hidden_size]
        let qkv = hidden_states.q_mat_mul(&self.wqkv).to_concrete();

        // Split into Q, K, V - each [batch, num_heads (or kv_heads), seq_len, head_dim]
        let query_states = split_heads(
            &qkv.narrow(2, 0, hidden_size).to_concrete(),
            self.num_heads,
            self.head_dim,
        );
        let key_states = split_heads(
            &qkv.narrow(2, hidden_size, hidden_size).to_concrete(),
            self.num_kv_heads,
            self.head_dim,
        );
        let value_states = split_heads(
            &qkv.narrow(2, 2 * hidden_size, hidden_size).to_concrete(),
            self.num_kv_heads,
            self.head_dim,
        );

        // Apply RoPE to Q and K
        let (query_states, key_states) = rope_cache.forward(&query_states, &key_states, 0);

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let [batch_size, _, seq_len, _] = query_states.shape();

        let attn_output = match window {
            // Local layer whose window actually constrains the sequence.
            Some(w) if seq_len > w + 1 => {
                // The band mask is a shared `[q, k]` tensor, but per-sample
                // padding lives on the key axis, so the combined mask is
                // logically `[batch, q, k]`. The fused flash-attention kernel
                // only accepts a 2D mask, so we fold the band and each sample's
                // padding into a per-element `QKMask` and run the batch as a
                // short loop (batch is typically 1 for single-text inference).
                let band = sliding_window_bias(seq_len, w, &self.device);
                let pad_bias = attention_mask.map(attention_mask_to_bias);
                let mut per_batch = Vec::with_capacity(batch_size);
                for b in 0..batch_size {
                    let q_b = query_states.narrow(0, b, 1).to_concrete();
                    let k_b = key_states.narrow(0, b, 1).to_concrete();
                    let v_b = value_states.narrow(0, b, 1).to_concrete();
                    let mask_b = match &pad_bias {
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
                Tensor::cat(per_batch, 0)
            }
            // Global attention, or a window wider than the sequence (no-op band):
            // batched flash attention with the per-sample padding mask.
            _ => {
                let mask = attention_mask.map(attention_mask_to_bias);
                query_states.flash_attention(
                    &key_states,
                    &value_states,
                    scale,
                    mask.as_ref().map(|m| (m, fusor::MaskKind::BatchKeyMask)),
                )
            }
        };

        // Merge heads and project output
        let attn_output = merge_heads(&attn_output);
        attn_output.q_mat_mul(&self.wo)
    }
}
