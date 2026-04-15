//! ModernBERT self-attention with RoPE and fused QKV.

use fusor::{Device, QMatrix, Result, RopeCache, Tensor, VarBuilder};

/// ModernBERT self-attention with fused QKV projection and RoPE.
pub struct ModernBertAttention {
    /// Fused QKV projection: [3 * hidden_size, hidden_size]
    wqkv: QMatrix,
    wo: QMatrix,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
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
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        rope_cache: &RopeCache,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        let hidden_size = self.num_heads * self.head_dim;

        // Compute fused QKV projection: [batch, seq_len, 3 * hidden_size]
        let qkv = hidden_states.q_mat_mul(&self.wqkv).to_concrete();

        // Split into Q, K, V - each [batch, num_heads (or kv_heads), seq_len, head_dim]
        use super::super::utils::split_heads;
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

        let mask = attention_mask.map(super::super::utils::attention_mask_to_bias);

        let attn_output = query_states.flash_attention(
            &key_states,
            &value_states,
            scale,
            mask.as_ref().map(|m| (m, fusor::MaskKind::BatchKeyMask)),
        );

        // Merge heads and project output
        let attn_output = super::super::utils::merge_heads(&attn_output);
        attn_output.q_mat_mul(&self.wo)
    }
}
