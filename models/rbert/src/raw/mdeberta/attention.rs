//! mDeBERTa disentangled self-attention.
//!
//! DeBERTa uses disentangled attention with three components:
//! - Content-to-Content (c2c): Standard attention between content vectors
//! - Content-to-Position (c2p): Attention from content to relative positions
//! - Position-to-Content (p2c): Attention from relative positions to content
//!
//! The attention score is: A = c2c + c2p + p2c

use fusor::layers::{LayerNorm, Linear};
use fusor::{Device, Result, Tensor, VarBuilder};

/// Relative position embeddings for disentangled attention.
pub struct RelativePositionEmbedding {
    /// Relative position embedding table [2*max_pos, hidden_size]
    embeddings: Tensor<2, f32>,
    /// LayerNorm applied to embeddings (norm_rel_ebd = "layer_norm" in DeBERTa)
    layer_norm: Option<LayerNorm<1, f32>>,
    /// Maximum relative positions (e.g., 256)
    max_relative_positions: usize,
}

impl RelativePositionEmbedding {
    /// Load with an already-loaded LayerNorm (avoids borrow issues)
    pub fn load_with_norm(
        device: &Device,
        vb: &mut VarBuilder,
        layer_norm: Option<LayerNorm<1, f32>>,
        max_relative_positions: usize,
    ) -> Result<Self> {
        let weight = vb.get("weight", device)?;
        let embeddings_raw: Tensor<2, f32> = weight.dequantize();

        // GGUF stores shape as [hidden_size, 2*max_pos] but we need [2*max_pos, hidden_size]
        let [dim0, dim1] = embeddings_raw.shape();
        let embeddings = if dim0 > dim1 {
            embeddings_raw.transpose(0, 1).to_concrete()
        } else {
            embeddings_raw
        };

        Ok(Self {
            embeddings,
            layer_norm,
            max_relative_positions,
        })
    }

    /// Apply log-bucket position encoding (matches Python make_log_bucket_position).
    /// positions close to 0 use linear indexing, far positions are log-bucketed.
    fn make_log_bucket_position(rel_pos: i32, bucket_size: i32, max_position: i32) -> i32 {
        let sign = if rel_pos > 0 {
            1
        } else if rel_pos < 0 {
            -1
        } else {
            0
        };
        let mid = bucket_size / 2;
        let abs_pos = if rel_pos < mid && rel_pos > -mid {
            mid - 1
        } else {
            rel_pos.abs()
        };
        if abs_pos <= mid {
            rel_pos
        } else {
            // log_pos = ceil(log(abs_pos/mid) / log((max-1)/mid) * (mid-1)) + mid
            let ratio = (abs_pos as f32) / (mid as f32);
            let max_ratio = ((max_position - 1) as f32) / (mid as f32);
            let log_pos = (ratio.ln() / max_ratio.ln() * ((mid - 1) as f32)).ceil() as i32 + mid;
            log_pos * sign
        }
    }

    /// Compute relative position indices for a sequence.
    /// Returns indices [seq_len, seq_len] where each entry is the relative position
    /// index into the embedding table.
    ///
    /// Matches Python: rel_pos_ids = q_ids[:,None] - k_ids[None,:] = i - j
    /// Then applies log bucketing with bucket_size=2*max_relative_positions (pos_ebd_size*2),
    /// max_position = 2*max_relative_positions... actually:
    /// - bucket_size = position_buckets = 256 (pos_ebd_size)
    /// - max_position = max_relative_positions = 512
    pub fn compute_relative_indices(&self, seq_len: usize, device: &Device) -> Tensor<2, u32> {
        // Python: bucket_size = position_buckets = 256, max_position = max_relative_positions = 512
        // att_span = pos_ebd_size = 256 (= bucket_size)
        // The position embedding table has 2*pos_ebd_size = 512 entries
        // After bucketing, rel_pos ranges in [-(pos_ebd_size), pos_ebd_size-1] approximately
        // c2p_pos = clamp(rel_pos + att_span, 0, 2*att_span-1) -> [0, 2*pos_ebd_size-1]
        let bucket_size = self.max_relative_positions as i32; // 256 (pos_ebd_size)
        let max_position = 2 * bucket_size; // 512 (2*pos_ebd_size = max_relative_positions)
        let att_span = bucket_size; // 256
        let num_positions = (2 * att_span) as i32; // 512

        let mut indices = vec![0u32; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                // Python: rel_pos = q - k = i - j
                let rel_pos = i as i32 - j as i32;
                // Apply log bucketing
                let bucketed = Self::make_log_bucket_position(rel_pos, bucket_size, max_position);
                // Shift to positive index: c2p_pos = clamp(bucketed + att_span, 0, 2*att_span-1)
                let idx = (bucketed + att_span).clamp(0, num_positions - 1) as u32;
                indices[i * seq_len + j] = idx;
            }
        }

        Tensor::new(device, &indices)
            .reshape([seq_len, seq_len])
            .to_concrete()
    }

    /// Get the raw relative position embedding table (normalized).
    /// Returns embeddings [2*max_pos, hidden_size]
    pub fn get_embeddings(&self) -> Tensor<2, f32> {
        // Apply LayerNorm to embeddings (like HuggingFace get_rel_embedding)
        if let Some(ref ln) = self.layer_norm {
            // Add batch dimension for LayerNorm: [num_positions, hidden_size] -> [1, num_positions, hidden_size]
            let emb_3d: Tensor<3, f32> = self.embeddings.unsqueeze(0).to_concrete();
            let normed = ln.forward(&emb_3d);
            // Remove batch dimension
            normed.squeeze(0).to_concrete()
        } else {
            self.embeddings.to_concrete()
        }
    }
}

/// mDeBERTa disentangled self-attention with shared key attention (share_att_key=True).
pub struct MDebertaAttention {
    query: Linear<f32>,
    key: Linear<f32>,
    value: Linear<f32>,
    output: Linear<f32>,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl MDebertaAttention {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let query = Linear::load(device, &mut vb.pp("query"))?;
        let key = Linear::load(device, &mut vb.pp("key"))?;
        let value = Linear::load(device, &mut vb.pp("value"))?;
        let output = Linear::load(device, &mut vb.pp("output"))?;

        // Scale factor for disentangled attention (3 components: c2c, c2p, p2c)
        // Python: scale = scaled_size_sqrt(query_layer, scale_factor) where scale_factor=3
        // This means sqrt(head_dim * 3)
        let scale = 1.0 / ((head_dim as f32) * 3.0).sqrt();

        Ok(Self {
            query,
            key,
            value,
            output,
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass with disentangled attention.
    ///
    /// # Arguments
    /// * `hidden_states` - Input [batch, seq_len, hidden_size]
    /// * `rel_pos_emb` - Relative position embedding table [2*max_pos, hidden_size]
    /// * `rel_pos_indices` - Relative position indices [seq_len, seq_len]
    /// * `attention_mask` - Optional attention mask [batch, seq_len]
    pub fn forward_with_indices(
        &self,
        hidden_states: &Tensor<3, f32>,
        rel_pos_emb: &Tensor<2, f32>,
        rel_pos_indices: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        use super::super::utils::split_heads;

        // Compute Q, K, V projections for content and reshape to
        // [batch, num_heads, seq_len, head_dim].
        let query = split_heads(
            &self.query.forward(hidden_states),
            self.num_heads,
            self.head_dim,
        );
        let key = split_heads(
            &self.key.forward(hidden_states),
            self.num_heads,
            self.head_dim,
        );
        let value = split_heads(
            &self.value.forward(hidden_states),
            self.num_heads,
            self.head_dim,
        );

        // === Content-to-Content attention ===
        let c2c_scores = query.mat_mul(&key.transpose(2, 3));

        // === Position attention with shared Q/K projections ===
        // rel_pos_emb: [2*max_pos, hidden_size] -> [1, 2*max_pos, hidden_size]
        let rel_emb_3d: Tensor<3, f32> = rel_pos_emb.unsqueeze(0).to_concrete();
        let pos_query = split_heads(
            &self.query.forward(&rel_emb_3d),
            self.num_heads,
            self.head_dim,
        );
        let pos_key = split_heads(
            &self.key.forward(&rel_emb_3d),
            self.num_heads,
            self.head_dim,
        );

        // === Content-to-Position attention ===
        // c2p = Q @ pos_key^T -> [batch, heads, seq, 2*max_pos]
        // Then gather based on relative positions
        let c2p_all = query.mat_mul(&pos_key.transpose(2, 3));
        let c2p_scores = self.gather_c2p(&c2p_all, rel_pos_indices);

        // === Position-to-Content attention ===
        // p2c = K @ pos_query^T -> [batch, heads, seq, 2*max_pos]
        // Then gather based on transposed relative positions
        let p2c_all = key.mat_mul(&pos_query.transpose(2, 3));
        let p2c_scores = self.gather_p2c(&p2c_all, rel_pos_indices);

        // Combine: attention = (c2c + c2p + p2c) * scale
        let attn_scores = c2c_scores
            .add_(&c2p_scores)
            .add_(&p2c_scores)
            .mul_scalar(self.scale);

        // Apply attention mask (broadcast bias to [batch, 1, 1, seq_len])
        let attn_scores = if let Some(mask) = attention_mask {
            let mask_bias = super::super::utils::attention_mask_to_bias(mask);
            let mask_bias_3d: Tensor<3, f32> = mask_bias.unsqueeze(1).to_concrete();
            let mask_bias_4d: Tensor<4, f32> = mask_bias_3d.unsqueeze(1).to_concrete();
            attn_scores.add_(&mask_bias_4d)
        } else {
            attn_scores
        };

        // Softmax
        let attn_probs = attn_scores.softmax_last_dim::<3>();

        // Apply attention to values and merge heads back to [batch, seq_len, hidden].
        let context = attn_probs.mat_mul(&value);
        let context = super::super::utils::merge_heads(&context);
        self.output.forward(&context)
    }

    /// Gather c2p attention scores based on relative position indices.
    ///
    /// Input: c2p_all [batch, heads, seq_len, 2*max_pos] - scores to all positions
    /// rel_pos_indices: [seq_len, seq_len] - index into position embeddings
    ///
    /// Output: [batch, heads, seq_len, seq_len] - gathered scores
    fn gather_c2p(
        &self,
        c2p_all: &Tensor<4, f32>,
        rel_pos_indices: &Tensor<2, u32>,
    ) -> Tensor<4, f32> {
        let [b_sz, num_heads, seq_len, _num_pos] = c2p_all.shape();
        let device = c2p_all.device();

        // Get data slices
        let c2p_data = pollster::block_on(c2p_all.clone().as_slice()).unwrap();
        let indices_data = pollster::block_on(rel_pos_indices.clone().as_slice()).unwrap();
        let c2p = c2p_data.as_slice();
        let indices = indices_data.as_slice();
        let num_pos = _num_pos;

        let mut gathered = vec![0.0f32; b_sz * num_heads * seq_len * seq_len];

        for b in 0..b_sz {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        // Index into c2p_all: [b, h, i, rel_pos[i,j]]
                        let rel_idx = indices[i * seq_len + j] as usize;
                        let c2p_idx = b * num_heads * seq_len * num_pos
                            + h * seq_len * num_pos
                            + i * num_pos
                            + rel_idx;
                        let out_idx = b * num_heads * seq_len * seq_len
                            + h * seq_len * seq_len
                            + i * seq_len
                            + j;
                        gathered[out_idx] = c2p[c2p_idx];
                    }
                }
            }
        }

        Tensor::new(&device, &gathered)
            .reshape([b_sz, num_heads, seq_len, seq_len])
            .to_concrete()
    }

    /// Gather p2c attention scores.
    ///
    /// Python derivation:
    /// - r_pos = relative_pos (since seq_q == seq_k)
    /// - p2c_pos[i,j] = clamp(-r_pos[i,j] + att_span, 0, 2*att_span-1)
    ///   = clamp(-(i-j) + att_span) = clamp((j-i) + att_span)
    /// - gather_out[b, m, n] = p2c_att[b, m, p2c_pos[m, n]]
    /// - final[b, i, j] = gather_out[b, j, i] (after transpose)
    ///   = p2c_att[b, j, p2c_pos[j, i]]
    ///   = p2c_att[b, j, clamp(i - j + att_span)]
    ///   = p2c_all[b, j, indices[i, j]]  (using indices[i,j] = bucketed(i-j) + att_span)
    fn gather_p2c(
        &self,
        p2c_all: &Tensor<4, f32>,
        rel_pos_indices: &Tensor<2, u32>,
    ) -> Tensor<4, f32> {
        let [b_sz, num_heads, seq_len, num_pos] = p2c_all.shape();
        let device = p2c_all.device();

        let p2c_data = pollster::block_on(p2c_all.clone().as_slice()).unwrap();
        let indices_data = pollster::block_on(rel_pos_indices.clone().as_slice()).unwrap();
        let p2c = p2c_data.as_slice();
        let indices = indices_data.as_slice();

        let mut gathered = vec![0.0f32; b_sz * num_heads * seq_len * seq_len];

        for b in 0..b_sz {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        // final[b, i, j] = p2c_all[b, j, indices[i, j]]
                        let rel_idx = (indices[i * seq_len + j] as usize).min(num_pos - 1);
                        let p2c_idx = b * num_heads * seq_len * num_pos
                            + h * seq_len * num_pos
                            + j * num_pos  // key dim = j
                            + rel_idx;
                        let out_idx = b * num_heads * seq_len * seq_len
                            + h * seq_len * seq_len
                            + i * seq_len
                            + j;
                        gathered[out_idx] = p2c[p2c_idx];
                    }
                }
            }
        }

        Tensor::new(&device, &gathered)
            .reshape([b_sz, num_heads, seq_len, seq_len])
            .to_concrete()
    }
}

/// Shared relative position embedding layer (used across all layers in DeBERTa).
pub struct DisentangledSelfAttention {
    attention: MDebertaAttention,
}

impl DisentangledSelfAttention {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let attention = MDebertaAttention::load(device, vb, num_heads, head_dim)?;
        Ok(Self { attention })
    }

    /// Forward with relative position indices and embedding table.
    pub fn forward_with_rel(
        &self,
        hidden_states: &Tensor<3, f32>,
        rel_pos_emb: &Tensor<2, f32>,
        rel_pos_indices: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        self.attention.forward_with_indices(
            hidden_states,
            rel_pos_emb,
            rel_pos_indices,
            attention_mask,
        )
    }
}
