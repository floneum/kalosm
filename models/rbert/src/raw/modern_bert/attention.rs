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
        let [b_sz, seq_len, _hidden_size] = hidden_states.shape();
        let hidden_size = self.num_heads * self.head_dim;

        // Compute fused QKV projection: [batch, seq_len, 3 * hidden_size]
        let qkv = hidden_states.q_mat_mul(&self.wqkv).to_concrete();

        // Split into Q, K, V - each [batch, seq_len, hidden_size]
        let query_states = qkv
            .narrow(2, 0, hidden_size)
            .reshape([b_sz, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete();

        let key_states = qkv
            .narrow(2, hidden_size, hidden_size)
            .reshape([b_sz, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete();

        let value_states = qkv
            .narrow(2, 2 * hidden_size, hidden_size)
            .reshape([b_sz, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete();

        // Apply RoPE to Q and K
        let (query_states, key_states) = rope_cache.forward(&query_states, &key_states, 0);

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Convert attention mask for flash attention if provided
        const MASK_NEG_VALUE: f32 = -10000.0;
        let mask: Option<Tensor<2, f32>> = attention_mask.map(|m| {
            let mask_f32: Tensor<2, f32> = m.cast();
            let zeros = mask_f32.zeros_like();
            let ones = (zeros + 1.0f32).to_concrete();
            ((ones - mask_f32) * MASK_NEG_VALUE).to_concrete()
        });

        let attn_output = query_states.flash_attention(
            &key_states,
            &value_states,
            scale,
            mask.as_ref().map(|m| (m, fusor::MaskKind::BatchKeyMask)),
        );

        // Reshape and project output
        let attn_output = attn_output.transpose(1, 2);
        let attn_output = attn_output
            .to_concrete()
            .reshape([b_sz, seq_len, hidden_size])
            .to_concrete();

        attn_output.q_mat_mul(&self.wo)
    }
}
