//! Scoring layer for span-label matching.

use fusor::Tensor;

/// Scorer for computing span-label similarity.
pub struct Scorer;

impl Scorer {
    /// Compute entity scores using dot product similarity.
    ///
    /// GLiNER uses raw dot product (not cosine similarity).
    /// The output logits are passed through sigmoid externally.
    ///
    /// # Arguments
    /// * `span_embeddings` - Span embeddings [batch, num_spans, hidden_dim]
    /// * `label_embeddings` - Label embeddings [num_labels, hidden_dim]
    ///
    /// # Returns
    /// Raw dot product scores [batch, num_spans, num_labels]
    pub fn forward(
        span_embeddings: &Tensor<3, f32>,
        label_embeddings: &Tensor<2, f32>,
    ) -> Tensor<3, f32> {
        let [batch_size, num_spans, hidden_dim] = span_embeddings.shape();
        let [num_labels, _] = label_embeddings.shape();

        // Flatten batch dimension for matmul
        let span_concrete = span_embeddings.to_concrete();
        let flat_spans = span_concrete
            .reshape([batch_size * num_spans, hidden_dim])
            .to_concrete();

        // Transpose labels: [hidden_dim, num_labels]
        let labels_t = label_embeddings.t();

        // Matmul: [batch * num_spans, hidden_dim] @ [hidden_dim, num_labels]
        // = [batch * num_spans, num_labels]
        let flat_logits = flat_spans.mat_mul(&labels_t);

        // Reshape to [batch, num_spans, num_labels]
        flat_logits
            .reshape([batch_size, num_spans, num_labels])
            .to_concrete()
    }
}

/// Apply sigmoid to raw scores in Rust (not on GPU).
///
/// # Arguments
/// * `logits` - Raw logit scores
///
/// # Returns
/// Probability scores (0.0 to 1.0)
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply sigmoid to a slice of logits.
pub fn apply_sigmoid(logits: &[f32]) -> Vec<f32> {
    logits.iter().map(|&x| sigmoid(x)).collect()
}
