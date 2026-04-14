//! Relation representation layer for adjacency matrix computation.
//!
//! Computes an adjacency matrix between entity spans to filter
//! candidate pairs for relation classification.

use fusor::layers::Linear;
use fusor::{Device, Result, Tensor, VarBuilder};

/// Relation representation layer - can be learned (bilinear) or simple dot-product.
pub enum RelationsRepLayer {
    /// Learned bilinear projection
    Bilinear(BilinearRelationsLayer),
    /// Simple dot-product similarity (no learned weights)
    DotProduct,
}

impl RelationsRepLayer {
    /// Load the relations layer from GGUF weights.
    /// Falls back to dot-product if weights don't exist.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        match BilinearRelationsLayer::load(device, vb) {
            Ok(bilinear) => Ok(Self::Bilinear(bilinear)),
            Err(_) => Ok(Self::DotProduct),
        }
    }

    /// Create a dot-product based relations layer (no learned weights).
    pub fn identity(_device: &Device, _hidden_size: usize) -> Self {
        Self::DotProduct
    }

    /// Compute adjacency matrix for entity spans.
    ///
    /// # Arguments
    /// * `entity_embeddings` - Entity span embeddings [batch, num_entities, hidden_size]
    ///
    /// # Returns
    /// Adjacency logits [batch, num_entities, num_entities] (apply sigmoid externally)
    pub fn forward(&self, entity_embeddings: &Tensor<3, f32>) -> Tensor<3, f32> {
        match self {
            Self::Bilinear(layer) => layer.forward(entity_embeddings),
            Self::DotProduct => {
                // Simple dot product: embeddings @ embeddings.T
                let entity_t = entity_embeddings.transpose(1, 2);
                entity_embeddings.mat_mul(&entity_t)
            }
        }
    }

    /// Apply sigmoid to logits (for use after forward).
    pub fn apply_sigmoid(logits: &[f32]) -> Vec<f32> {
        logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    /// Filter entity pairs based on adjacency threshold.
    ///
    /// # Arguments
    /// * `adjacency_scores` - Adjacency matrix [batch, num_entities, num_entities]
    /// * `threshold` - Minimum score for a pair to be considered
    ///
    /// # Returns
    /// Vector of (batch_idx, head_idx, tail_idx, score) tuples for pairs above threshold
    pub async fn filter_pairs(
        &self,
        adjacency_scores: &Tensor<3, f32>,
        threshold: f32,
    ) -> Result<Vec<(usize, usize, usize, f32)>> {
        let [batch_size, num_entities, _] = adjacency_scores.shape();

        let scores_slice = adjacency_scores.clone().as_slice().await?;
        let scores_data = scores_slice.as_slice();

        let mut pairs = Vec::new();
        for b in 0..batch_size {
            for i in 0..num_entities {
                for j in 0..num_entities {
                    if i == j {
                        continue; // Skip self-relations
                    }
                    let idx = b * num_entities * num_entities + i * num_entities + j;
                    let score = scores_data[idx];
                    if score >= threshold {
                        pairs.push((b, i, j, score));
                    }
                }
            }
        }

        Ok(pairs)
    }
}

/// Learned bilinear relation representation layer.
///
/// Computes adjacency scores between entity pairs:
/// `adj_score[i,j] = sigmoid(entity_i @ W @ entity_j.T)`
pub struct BilinearRelationsLayer {
    /// Bilinear projection weight [hidden_size, hidden_size]
    projection: Linear<f32>,
}

impl BilinearRelationsLayer {
    /// Load from GGUF weights.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let projection = Linear::load(device, &mut vb.pp("projection"))?;
        Ok(Self { projection })
    }

    /// Compute adjacency matrix for entity spans.
    pub fn forward(&self, entity_embeddings: &Tensor<3, f32>) -> Tensor<3, f32> {
        // Project entity embeddings: [batch, num_entities, hidden_size]
        let projected = self.projection.forward(entity_embeddings);

        // Compute bilinear scores: projected @ entity_embeddings.T
        // [batch, num_entities, hidden_size] @ [batch, hidden_size, num_entities]
        // = [batch, num_entities, num_entities]
        let entity_t = entity_embeddings.transpose(1, 2);
        projected.mat_mul(&entity_t)
    }
}
