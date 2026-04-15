//! mDeBERTa-v3 encoder model.

use fusor::layers::{Embedding, LayerNorm, Linear};
use fusor::{Device, Result, Tensor, VarBuilder};

use super::attention::RelativePositionEmbedding;
use super::config::MDebertaConfig;
use super::layer::MDebertaLayer;

/// A raw synchronous mDeBERTa-v3 encoder model. This is a bidirectional
/// transformer encoder using disentangled attention with relative position
/// embeddings (DeBERTa-v3 architecture).
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
    span: tracing::Span,
}

impl MDebertaModel {
    /// Load mDeBERTa from GGUF weights.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let config = MDebertaConfig::from_gguf(vb)?;

        let token_embeddings = Embedding::load(device, &mut vb.pp("token_embd"))?;
        let embedding_norm = LayerNorm::load(device, &mut vb.pp("embd_norm"), config.norm_eps)?;

        // The `output_norm` tensor in GGUF is the LayerNorm for the relative
        // position embeddings. Load it first to avoid borrow issues.
        let rel_pos_norm = LayerNorm::load(device, &mut vb.pp("output_norm"), config.norm_eps).ok();
        let rel_pos_embedding = RelativePositionEmbedding::load_with_norm(
            device,
            &mut vb.pp("rel_pos_embd"),
            rel_pos_norm,
            config.max_relative_positions,
        )?;

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

        Ok(Self {
            token_embeddings,
            embedding_norm,
            rel_pos_embedding,
            layers,
            output_proj,
            device: device.clone(),
            config,
            span: tracing::span!(tracing::Level::TRACE, "mdeberta"),
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
        let _enter = self.span.enter();
        let [_batch_size, seq_len] = input_ids.shape();

        let [b_sz, _] = input_ids.shape();
        let hidden_states = self.token_embeddings.forward(input_ids);
        let mut hidden_states = self.embedding_norm.forward(&hidden_states);

        // Compute the flat gather indices once per forward; every layer shares them.
        let gather_idx = self.rel_pos_embedding.compute_gather_indices(
            b_sz,
            self.config.num_heads,
            seq_len,
            &self.device,
        );
        let rel_pos_emb = self.rel_pos_embedding.get_embeddings();

        for layer in &self.layers {
            hidden_states =
                layer.forward_with_rel(&hidden_states, &rel_pos_emb, &gather_idx, attention_mask);
        }

        if let Some(ref proj) = self.output_proj {
            hidden_states = proj.forward(&hidden_states);
        }

        hidden_states
    }

    #[doc(hidden)]
    pub fn debug_after_embedding_norm(&self, input_ids: &Tensor<2, u32>) -> Tensor<3, f32> {
        let hidden_states = self.token_embeddings.forward(input_ids);
        self.embedding_norm.forward(&hidden_states)
    }

    #[doc(hidden)]
    pub fn debug_first_layer_output(
        &self,
        hidden_states: &Tensor<3, f32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        let [b_sz, seq_len, _] = hidden_states.shape();
        let gather_idx = self.rel_pos_embedding.compute_gather_indices(
            b_sz,
            self.config.num_heads,
            seq_len,
            &self.device,
        );
        let rel_pos_emb = self.rel_pos_embedding.get_embeddings();
        self.layers[0].forward_with_rel(hidden_states, &rel_pos_emb, &gather_idx, attention_mask)
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
