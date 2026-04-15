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

/// Precomputed flat index tensors for the c2p and p2c gathers, valid for a
/// single `(b_sz, num_heads, seq_len)` combination. Built once per forward in
/// [`MDebertaModel::forward`] and threaded through every layer.
pub struct GatherIndices {
    /// Flat source indices for c2p, shape `[b_sz * num_heads * seq_len * seq_len]`.
    pub(crate) c2p: Tensor<1, u32>,
    /// Flat source indices for p2c, shape `[b_sz * num_heads * seq_len * seq_len]`.
    pub(crate) p2c: Tensor<1, u32>,
    pub(crate) b_sz: usize,
    pub(crate) num_heads: usize,
    pub(crate) seq_len: usize,
}

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

    /// Number of entries (`2 * max_relative_positions`) in the relative
    /// position embedding table — this is the per-head "position dimension" of
    /// the `c2p_all` / `p2c_all` attention scores before gathering.
    pub fn num_positions(&self) -> usize {
        self.embeddings.shape()[0]
    }

    /// Build the flat index tensors used to gather `c2p_all` and `p2c_all`
    /// along the last two dims in a single on-device `index_select`.
    ///
    /// The gather semantics are:
    ///   c2p_out[b, h, i, j] = c2p_all[b, h, i, indices[i, j]]
    ///   p2c_out[b, h, i, j] = p2c_all[b, h, j, indices[i, j]]
    ///
    /// `indices[i, j]` is the DeBERTa log-bucketed relative position index. We
    /// bake the `(b, h, i|j)` outer offsets into a single 1D `u32` tensor of
    /// length `b_sz * num_heads * seq_len * seq_len`, so each gather becomes:
    ///   source.reshape([b*h*s*p]).index_select(0, flat_idx).reshape([b,h,s,s])
    pub fn compute_gather_indices(
        &self,
        b_sz: usize,
        num_heads: usize,
        seq_len: usize,
        device: &Device,
    ) -> GatherIndices {
        let num_pos = self.num_positions();
        let bucket_size = self.max_relative_positions as i32;
        let max_position = 2 * bucket_size;
        let att_span = bucket_size;
        let num_positions_i = (2 * att_span) as i32;

        // Raw relative-position indices, [seq_len, seq_len].
        let mut rel = vec![0u32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let rel_pos = i as i32 - j as i32;
                let bucketed = Self::make_log_bucket_position(rel_pos, bucket_size, max_position);
                let idx = (bucketed + att_span).clamp(0, num_positions_i - 1) as u32;
                rel[i * seq_len + j] = idx;
            }
        }

        let total = b_sz * num_heads * seq_len * seq_len;
        let mut c2p = vec![0u32; total];
        let mut p2c = vec![0u32; total];
        for b in 0..b_sz {
            for h in 0..num_heads {
                let bh_offset = ((b * num_heads + h) * seq_len) * num_pos;
                for i in 0..seq_len {
                    let row_offset_c2p = bh_offset + i * num_pos;
                    for j in 0..seq_len {
                        let rel_idx = rel[i * seq_len + j] as usize;
                        // c2p: key dim = i
                        c2p[((b * num_heads + h) * seq_len + i) * seq_len + j] =
                            (row_offset_c2p + rel_idx.min(num_pos - 1)) as u32;
                        // p2c: key dim = j (different row offset)
                        let row_offset_p2c = bh_offset + j * num_pos;
                        p2c[((b * num_heads + h) * seq_len + i) * seq_len + j] =
                            (row_offset_p2c + rel_idx.min(num_pos - 1)) as u32;
                    }
                }
            }
        }

        GatherIndices {
            c2p: Tensor::new(device, &c2p),
            p2c: Tensor::new(device, &p2c),
            b_sz,
            num_heads,
            seq_len,
        }
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
    /// * `gather_idx` - Precomputed flat indices for the c2p / p2c gathers.
    /// * `attention_mask` - Optional attention mask [batch, seq_len]
    pub fn forward_with_indices(
        &self,
        hidden_states: &Tensor<3, f32>,
        rel_pos_emb: &Tensor<2, f32>,
        gather_idx: &GatherIndices,
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
        let [batch_size, _, _, _] = query.shape();

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
        let num_relative_positions = pos_query.shape()[2];
        let pos_query = pos_query
            .broadcast_as([
                batch_size,
                self.num_heads,
                num_relative_positions,
                self.head_dim,
            ])
            .to_concrete();
        let pos_key = pos_key
            .broadcast_as([
                batch_size,
                self.num_heads,
                num_relative_positions,
                self.head_dim,
            ])
            .to_concrete();

        // === Content-to-Position attention ===
        // c2p = Q @ pos_key^T -> [batch, heads, seq, 2*max_pos]
        // Then gather based on relative positions
        let c2p_all = query.mat_mul(&pos_key.transpose(2, 3));
        let c2p_scores = gather_by_flat_index(&c2p_all, gather_idx, &gather_idx.c2p);

        // === Position-to-Content attention ===
        // p2c = K @ pos_query^T -> [batch, heads, seq, 2*max_pos]
        // Then gather based on transposed relative positions
        let p2c_all = key.mat_mul(&pos_query.transpose(2, 3));
        let p2c_scores = gather_by_flat_index(&p2c_all, gather_idx, &gather_idx.p2c);

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
}

/// On-device gather used by both c2p and p2c. `flat_idx` encodes the full
/// `((b*H + h)*S + i_or_j) * P + rel[i, j]` source offset so that the gather
/// reduces to flatten → `index_select` → reshape.
fn gather_by_flat_index(
    src: &Tensor<4, f32>,
    shape: &GatherIndices,
    flat_idx: &Tensor<1, u32>,
) -> Tensor<4, f32> {
    let [b, h, s, p] = src.shape();
    debug_assert_eq!(b, shape.b_sz);
    debug_assert_eq!(h, shape.num_heads);
    debug_assert_eq!(s, shape.seq_len);
    src.reshape([b * h * s * p])
        .index_select(0, flat_idx)
        .reshape([b, h, s, s])
        .to_concrete()
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

    /// Forward with precomputed gather indices and relative embedding table.
    pub fn forward_with_rel(
        &self,
        hidden_states: &Tensor<3, f32>,
        rel_pos_emb: &Tensor<2, f32>,
        gather_idx: &GatherIndices,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        self.attention
            .forward_with_indices(hidden_states, rel_pos_emb, gather_idx, attention_mask)
    }
}
