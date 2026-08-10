use fusor2::cache::MaskKind;
use fusor2::composite::attention::{attention, attention_masked};
use fusor2::composite::rope::rope_normal_pair_fused;
use fusor2::device::Device;
use fusor2::layers::RmsNorm;
use fusor2::tensor::Dyn as Tensor;
use fusor2::{Dim, Result, VarBuilder};

use super::model::QwenRope;
use super::QLinear;
use crate::raw::additive_key_mask;

/// Qwen self-attention with separate Q/K/V projections and RoPE
pub struct QwenSelfAttention {
    wq: QLinear,
    wk: QLinear,
    wv: QLinear,
    wo: QLinear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QwenSelfAttention {
    pub fn load(
        device: &Device,
        vb: &VarBuilder,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Self> {
        let wq = QLinear::load(vb, device, "attn_q.weight")?;
        let wk = QLinear::load(vb, device, "attn_k.weight")?;
        let wv = QLinear::load(vb, device, "attn_v.weight")?;
        let wo = QLinear::load(vb, device, "attn_output.weight")?;

        // Optional Q/K normalization (some Qwen models have this)
        let graph = device.graph().handle();
        let q_norm = vb
            .pp("attn_q_norm")
            .contains_key("weight")
            .then(|| RmsNorm::load(&vb.pp("attn_q_norm"), graph, eps))
            .transpose()?;
        let k_norm = vb
            .pp("attn_k_norm")
            .contains_key("weight")
            .then(|| RmsNorm::load(&vb.pp("attn_k_norm"), graph, eps))
            .transpose()?;

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

    /// `[B, L, heads * head_dim] -> [B, heads, L, head_dim]`.
    fn split_heads(&self, x: &Tensor, heads: usize) -> Result<Tensor> {
        let shape = x.shape();
        x.reshape_dims(&[
            shape[0],
            shape[1],
            Dim::Const(heads as u64),
            Dim::Const(self.head_dim as u64),
        ])?
        .transpose(1, 2)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &QwenRope,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let (b_sz, seq_len) = (shape[0], shape[1]);

        // Compute Q, K, V projections
        let mut query_states =
            self.split_heads(&self.wq.forward(hidden_states)?, self.num_heads)?;
        let mut key_states = self.split_heads(&self.wk.forward(hidden_states)?, self.num_kv_heads)?;
        let value_states = self.split_heads(&self.wv.forward(hidden_states)?, self.num_kv_heads)?;

        // Apply optional Q/K normalization
        if let Some(q_norm) = &self.q_norm {
            query_states = q_norm.forward(&query_states)?;
        }
        if let Some(k_norm) = &self.k_norm {
            key_states = k_norm.forward(&key_states)?;
        }

        // Apply RoPE to Q and K (Qwen uses the non-interleaved half layout)
        let (query_states, key_states) =
            rope_normal_pair_fused(&query_states, &key_states, &rope.cos, &rope.sin, 0)?;

        // Scaled dot-product attention. Grouped-query attention is handled
        // structurally by the composite: no K/V head expansion here.
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn_output = match attention_mask {
            Some(mask) => {
                let mask = additive_key_mask(mask)?;
                attention_masked(
                    &query_states,
                    &key_states,
                    &value_states,
                    MaskKind::BatchKeyMask,
                    Some(&mask),
                    Some(scale),
                )?
            }
            None => attention(
                &query_states,
                &key_states,
                &value_states,
                MaskKind::None,
                Some(scale),
            )?,
        };

        // Reshape and project output
        let hidden_size = self.num_heads * self.head_dim;
        let attn_output = attn_output.transpose(1, 2)?.reshape_dims(&[
            b_sz,
            seq_len,
            Dim::Const(hidden_size as u64),
        ])?;

        self.wo.forward(&attn_output)
    }
}
