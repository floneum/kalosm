//! mDeBERTa-v3 encoder model.

#[cfg(debug_assertions)]
use pollster;

use fusor::layers::{Embedding, LayerNorm, Linear};
use fusor::{Device, Result, Tensor, VarBuilder};

use super::attention::RelativePositionEmbedding;
use super::config::MDebertaConfig;
use super::layer::MDebertaLayer;

/// mDeBERTa-v3 encoder model for GLiNER-RelEx.
///
/// This is a bidirectional transformer encoder using disentangled attention
/// with relative position embeddings.
pub struct MDebertaModel {
    /// Token embeddings
    token_embeddings: Embedding<f32>,
    /// Embedding LayerNorm
    embedding_norm: LayerNorm<1, f32>,
    /// Relative position embeddings (shared across layers, with LayerNorm)
    rel_pos_embedding: RelativePositionEmbedding,
    /// Transformer layers
    layers: Vec<MDebertaLayer>,
    /// Optional encoder output projection (used by the `large` variants to
    /// map the encoder's 1024-dim hidden state down to the 768-dim space
    /// expected by the downstream heads). Absent on base/multi variants.
    output_proj: Option<Linear<f32>>,
    /// Device
    device: Device,
    /// Configuration
    config: MDebertaConfig,
}

impl MDebertaModel {
    /// Load mDeBERTa from GGUF weights.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let config = MDebertaConfig::from_gguf(vb)?;

        // Load token embeddings
        let token_embeddings = Embedding::load(device, &mut vb.pp("token_embd"))?;

        // Load embedding LayerNorm
        let embedding_norm = LayerNorm::load(device, &mut vb.pp("embd_norm"), config.norm_eps)?;

        // Load relative position embeddings with LayerNorm
        // The output_norm in GGUF is the LayerNorm for relative position embeddings
        // Load norm first to avoid borrow issues
        let rel_pos_norm = LayerNorm::load(device, &mut vb.pp("output_norm"), config.norm_eps).ok();
        let rel_pos_embedding = RelativePositionEmbedding::load_with_norm(
            device,
            &mut vb.pp("rel_pos_embd"),
            rel_pos_norm,
            config.max_relative_positions,
        )?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = MDebertaLayer::load(
                device,
                &mut vb.pp(format!("blk.{i}")),
                config.num_heads,
                config.head_dimension,
                config.norm_eps,
            )?;
            layers.push(layer);
        }

        // Optional post-encoder projection (only present on the large variants,
        // which use DeBERTa-v3-large at 1024-dim and project down to 768).
        let output_proj = Linear::load(device, &mut vb.pp("output_proj")).ok();

        #[cfg(debug_assertions)]
        if let Some(ref p) = output_proj {
            eprintln!(
                "[DEBUG] Encoder output projection loaded: {} -> {}",
                p.in_features(),
                p.out_features()
            );
        }

        Ok(Self {
            token_embeddings,
            embedding_norm,
            rel_pos_embedding,
            layers,
            output_proj,
            device: device.clone(),
            config,
        })
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    /// * `attention_mask` - Optional attention mask [batch, seq_len]
    ///
    /// # Returns
    /// Hidden states [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        input_ids: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        let [_batch_size, seq_len] = input_ids.shape();

        // Get token embeddings
        let mut hidden_states = self.token_embeddings.forward(input_ids);

        #[cfg(debug_assertions)]
        {
            let data = pollster::block_on(hidden_states.clone().as_slice()).unwrap();
            let slice = data.as_slice();
            let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
            let std: f32 = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / slice.len() as f32).sqrt();
            eprintln!("[DEBUG] After token_embeddings: mean={:.6}, std={:.6}", mean, std);
        }

        // Apply embedding LayerNorm
        hidden_states = self.embedding_norm.forward(&hidden_states);

        #[cfg(debug_assertions)]
        {
            let data = pollster::block_on(hidden_states.clone().as_slice()).unwrap();
            let slice = data.as_slice();
            let hidden_size = self.config.hidden_size;
            let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
            let std: f32 = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / slice.len() as f32).sqrt();
            eprintln!("[DEBUG] After embedding_norm: mean={:.6}, std={:.6}", mean, std);
            // Print raw embeddings at <<ENT>> positions (1, 3, 5) and others
            eprintln!("[DEBUG] Raw embeddings at positions (first 5 values):");
            for pos in [0, 1, 2, 3, 4, 5, 10, 17] {
                if pos < seq_len {
                    let start = pos * hidden_size;
                    let vals: Vec<f32> = (0..5).map(|i| slice[start + i]).collect();
                    eprintln!("  pos {}: {:?}", pos, vals);
                }
            }
        }

        // Compute relative position indices and get embedding table
        let rel_indices = self.rel_pos_embedding.compute_relative_indices(seq_len, &self.device);
        let rel_pos_emb = self.rel_pos_embedding.get_embeddings();

        // Pass through transformer layers with proper position attention
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_with_rel(&hidden_states, &rel_pos_emb, &rel_indices, attention_mask);

            #[cfg(debug_assertions)]
            if i == 0 || i == 11 {
                let data = pollster::block_on(hidden_states.clone().as_slice()).unwrap();
                let slice = data.as_slice();
                let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
                let std: f32 = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / slice.len() as f32).sqrt();
                eprintln!("[DEBUG] After layer {}: mean={:.6}, std={:.6}", i, mean, std);
            }
        }

        #[cfg(debug_assertions)]
        {
            let data = pollster::block_on(hidden_states.clone().as_slice()).unwrap();
            let slice = data.as_slice();
            let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
            let std: f32 = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / slice.len() as f32).sqrt();
            eprintln!("[DEBUG] Encoder output (pre-projection): mean={:.6}, std={:.6}", mean, std);
        }

        // Apply optional post-encoder projection (large variants).
        if let Some(ref proj) = self.output_proj {
            hidden_states = proj.forward(&hidden_states);

            #[cfg(debug_assertions)]
            {
                let data = pollster::block_on(hidden_states.clone().as_slice()).unwrap();
                let slice = data.as_slice();
                let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
                let std: f32 = (slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / slice.len() as f32).sqrt();
                eprintln!(
                    "[DEBUG] Encoder output (post-projection): mean={:.6}, std={:.6}",
                    mean, std
                );
            }
        }

        hidden_states
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.hidden_size
    }

    /// Get the maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.config.context_length
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the configuration.
    pub fn config(&self) -> &MDebertaConfig {
        &self.config
    }
}
