use fusor::layers::RmsNorm;
use fusor::{Device, QMatrix, Result, RopeCache, Tensor, VarBuilder};

/// Qwen self-attention with separate Q/K/V projections and RoPE
pub struct QwenSelfAttention {
    wq: QMatrix,
    wk: QMatrix,
    wv: QMatrix,
    wo: QMatrix,
    q_norm: Option<RmsNorm<1, f32>>,
    k_norm: Option<RmsNorm<1, f32>>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QwenSelfAttention {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Self> {
        let wq = vb.get("attn_q.weight", device)?;
        let wk = vb.get("attn_k.weight", device)?;
        let wv = vb.get("attn_v.weight", device)?;
        let wo = vb.get("attn_output.weight", device)?;

        // Optional Q/K normalization (some Qwen models have this)
        let q_norm = RmsNorm::load(device, &mut vb.pp("attn_q_norm"), eps).ok();
        let k_norm = RmsNorm::load(device, &mut vb.pp("attn_k_norm"), eps).ok();

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
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

        // Compute Q, K, V projections
        let mut query_states = hidden_states
            .q_mat_mul(&self.wq)
            .reshape([b_sz, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete();

        let mut key_states = hidden_states
            .q_mat_mul(&self.wk)
            .reshape([b_sz, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete();

        let value_states = hidden_states
            .q_mat_mul(&self.wv)
            .reshape([b_sz, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2)
            .to_concrete();

        // Apply optional Q/K normalization
        if let Some(ref q_norm) = self.q_norm {
            query_states = q_norm.forward_4d(&query_states);
        }
        if let Some(ref k_norm) = self.k_norm {
            key_states = k_norm.forward_4d(&key_states);
        }

        // Apply RoPE to Q and K
        let (query_states, key_states) = rope_cache.forward(&query_states, &key_states, 0);

        // Scaled dot-product attention
        let hidden_size = self.num_heads * self.head_dim;
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Convert attention mask for flash attention if provided
        // The mask should be [b_sz, seq_len] where 1 = valid, 0 = pad
        // Flash attention expects None for no mask, or a mask tensor
        // Note: We use a large negative value instead of NEG_INFINITY because
        // WGSL shaders don't support inf literals. -10000 is enough to effectively
        // zero out masked positions after softmax.
        const MASK_NEG_VALUE: f32 = -10000.0;
        let mask: Option<Tensor<2, f32>> = attention_mask.map(|m| {
            // Convert u32 mask to f32
            // 1 (valid) -> 0.0, 0 (pad) -> large negative value
            let mask_f32: Tensor<2, f32> = m.cast();
            // Create ones by adding 1 to zeros
            let zeros = mask_f32.zeros_like();
            let ones = (zeros + 1.0f32).to_concrete();
            // (1 - mask) * large_neg gives: valid=0, pad=large_neg
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
