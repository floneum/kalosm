//! Span representation layer.
//!
//! The actual GLiNER architecture uses:
//! - project_start: 2-layer FFN for start word
//! - project_end: 2-layer FFN for end word
//! - out_project: 2-layer FFN for combined (start + end) representation

use fusor::layers::Linear;
use fusor::{Device, Result, Tensor, VarBuilder};

/// Span representation layer.
///
/// Creates span embeddings by projecting start and end word embeddings
/// separately, then combining them through an output projection.
pub struct SpanLayer {
    /// Project start word: [hidden_dim] -> [hidden_dim]
    start_fc1: Linear<f32>,
    start_fc2: Linear<f32>,
    /// Project end word: [hidden_dim] -> [hidden_dim]
    end_fc1: Linear<f32>,
    end_fc2: Linear<f32>,
    /// Output projection: [2 * hidden_dim] -> [hidden_dim]
    out_fc1: Linear<f32>,
    out_fc2: Linear<f32>,
}

impl SpanLayer {
    /// Load span layer from GGUF weights.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        // Try different weight naming conventions
        let start_fc1 = Linear::load(device, &mut vb.pp("span.start_fc1")).or_else(|_| {
            Linear::load(
                device,
                &mut vb.pp("span_rep_layer.span_rep_layer.project_start.0"),
            )
        })?;
        let start_fc2 = Linear::load(device, &mut vb.pp("span.start_fc2")).or_else(|_| {
            Linear::load(
                device,
                &mut vb.pp("span_rep_layer.span_rep_layer.project_start.3"),
            )
        })?;

        let end_fc1 = Linear::load(device, &mut vb.pp("span.end_fc1")).or_else(|_| {
            Linear::load(
                device,
                &mut vb.pp("span_rep_layer.span_rep_layer.project_end.0"),
            )
        })?;
        let end_fc2 = Linear::load(device, &mut vb.pp("span.end_fc2")).or_else(|_| {
            Linear::load(
                device,
                &mut vb.pp("span_rep_layer.span_rep_layer.project_end.3"),
            )
        })?;

        let out_fc1 = Linear::load(device, &mut vb.pp("span.out_fc1")).or_else(|_| {
            Linear::load(
                device,
                &mut vb.pp("span_rep_layer.span_rep_layer.out_project.0"),
            )
        })?;
        let out_fc2 = Linear::load(device, &mut vb.pp("span.out_fc2")).or_else(|_| {
            Linear::load(
                device,
                &mut vb.pp("span_rep_layer.span_rep_layer.out_project.3"),
            )
        })?;

        Ok(Self {
            start_fc1,
            start_fc2,
            end_fc1,
            end_fc2,
            out_fc1,
            out_fc2,
        })
    }

    /// Compute span representations for a batch of per-item span lists.
    ///
    /// Returns:
    /// - flattened span embeddings in batch-major order
    /// - one count per batch item so the caller can slice the flattened output
    pub fn forward_for_spans_batched(
        &self,
        word_embeddings: &Tensor<3, f32>,
        spans_per_batch: &[Vec<(usize, usize)>],
        device: &Device,
    ) -> (Tensor<2, f32>, Vec<usize>) {
        let [batch_size, num_words, hidden_dim] = word_embeddings.shape();
        assert_eq!(
            batch_size,
            spans_per_batch.len(),
            "spans_per_batch must match batch size"
        );

        let span_counts: Vec<usize> = spans_per_batch.iter().map(Vec::len).collect();
        let total_spans: usize = span_counts.iter().sum();
        if total_spans == 0 {
            // No spans: return an empty (0-row) tensor so the row count matches
            // the all-zero `span_counts`, consistent with the non-empty path.
            return (Tensor::zeros(device, [0, hidden_dim]), span_counts);
        }

        let start_rep = self
            .start_fc2
            .forward(&self.start_fc1.forward(word_embeddings).relu());
        let end_rep = self
            .end_fc2
            .forward(&self.end_fc1.forward(word_embeddings).relu());

        let start_rep_flat = start_rep
            .to_concrete()
            .reshape([batch_size * num_words, hidden_dim])
            .to_concrete();
        let end_rep_flat = end_rep
            .to_concrete()
            .reshape([batch_size * num_words, hidden_dim])
            .to_concrete();

        let mut start_offset_indices: Vec<u32> = Vec::with_capacity(total_spans);
        let mut end_offset_indices: Vec<u32> = Vec::with_capacity(total_spans);
        for (batch_idx, spans) in spans_per_batch.iter().enumerate() {
            let offset = (batch_idx * num_words) as u32;
            for &(start, end) in spans {
                start_offset_indices.push(start as u32 + offset);
                end_offset_indices.push(end as u32 + offset);
            }
        }

        let start_idx_tensor = Tensor::new(device, &start_offset_indices);
        let end_idx_tensor = Tensor::new(device, &end_offset_indices);

        let start_gathered = start_rep_flat.index_select(0, &start_idx_tensor);
        let end_gathered = end_rep_flat.index_select(0, &end_idx_tensor);
        let combined = Tensor::cat([start_gathered, end_gathered], 1)
            .reshape([1, total_spans, hidden_dim * 2])
            .to_concrete()
            .relu();
        let hidden = self.out_fc1.forward(&combined).relu();
        let out = self
            .out_fc2
            .forward(&hidden)
            .reshape([total_spans, hidden_dim])
            .to_concrete();

        (out, span_counts)
    }
}
