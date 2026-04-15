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
    /// Maximum span width
    max_width: usize,
}

impl SpanLayer {
    /// Load span layer from GGUF weights.
    pub fn load(device: &Device, vb: &mut VarBuilder, max_width: usize) -> Result<Self> {
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
            max_width,
        })
    }

    /// Enumerate all valid spans up to max_width.
    ///
    /// Returns Vec of (start_word, end_word) pairs.
    pub fn enumerate_spans(&self, num_words: usize) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        for start in 0..num_words {
            for width in 1..=self.max_width.min(num_words - start) {
                let end = start + width - 1;
                spans.push((start, end));
            }
        }
        spans
    }

    /// Generate span representations from word embeddings.
    ///
    /// # Arguments
    /// * `word_embeddings` - Word embeddings [batch, num_words, hidden_dim]
    /// * `device` - Device for output tensors
    ///
    /// # Returns
    /// * Span embeddings [batch, num_spans, hidden_dim]
    /// * Span indices (start_word, end_word) for each span
    pub fn forward(
        &self,
        word_embeddings: &Tensor<3, f32>,
        device: &Device,
    ) -> (Tensor<3, f32>, Vec<(usize, usize)>) {
        let shape = word_embeddings.shape();
        let batch_size = shape[0];
        let num_words = shape[1];
        let hidden_dim = shape[2];

        // Enumerate all valid spans
        let span_indices = self.enumerate_spans(num_words);
        let num_spans = span_indices.len();

        if num_spans == 0 {
            // Return empty tensor if no spans
            let empty = Tensor::zeros(device, [batch_size, 1, hidden_dim]);
            return (empty, vec![(0, 0)]);
        }

        // Build start and end embeddings for all spans
        let (start_emb, end_emb) =
            self.gather_span_embeddings(word_embeddings, &span_indices, device);

        // Project start embeddings: [batch, num_spans, hidden_dim]
        // create_projection_layer uses: Linear -> ReLU -> Dropout -> Linear
        let start_projected = self
            .start_fc2
            .forward(&self.start_fc1.forward(&start_emb).relu());

        // Project end embeddings: [batch, num_spans, hidden_dim]
        let end_projected = self.end_fc2.forward(&self.end_fc1.forward(&end_emb).relu());

        // Concatenate: [batch, num_spans, 2 * hidden_dim]
        // Python does: cat([start, end]).relu() before out_project
        let combined = Tensor::cat(
            [start_projected.to_concrete(), end_projected.to_concrete()],
            2,
        )
        .relu();

        // Output projection: [batch, num_spans, hidden_dim]
        let span_embeddings = self
            .out_fc2
            .forward(&self.out_fc1.forward(&combined).relu());

        (span_embeddings, span_indices)
    }

    /// Compute span representations for specific (start, end) word positions.
    ///
    /// Matches Python TokenMarker forward:
    /// 1. project_start(h) -> start_rep
    /// 2. project_end(h) -> end_rep
    /// 3. gather at span positions
    /// 4. cat + relu
    /// 5. out_project
    ///
    /// # Arguments
    /// * `word_embeddings` - Word embeddings [batch=1, num_words, hidden]
    /// * `spans` - List of (start_word, end_word) pairs
    ///
    /// # Returns
    /// Span embeddings [num_spans, hidden]
    pub fn forward_for_spans(
        &self,
        word_embeddings: &Tensor<3, f32>,
        spans: &[(usize, usize)],
        device: &Device,
    ) -> Tensor<2, f32> {
        let (batched, counts) =
            self.forward_for_spans_batched(word_embeddings, &[spans.to_vec()], device);
        let _count = counts.first().copied().unwrap_or(0);
        batched
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
            return (Tensor::zeros(device, [1, hidden_dim]), span_counts);
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

    fn gather_span_embeddings(
        &self,
        word_embeddings: &Tensor<3, f32>,
        span_indices: &[(usize, usize)],
        device: &Device,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        let shape = word_embeddings.shape();
        let batch_size = shape[0];
        let num_words = shape[1];
        let hidden_dim = shape[2];
        let num_spans = span_indices.len();

        // Create index tensors for gathering
        let start_indices: Vec<u32> = span_indices.iter().map(|(s, _)| *s as u32).collect();
        let end_indices: Vec<u32> = span_indices.iter().map(|(_, e)| *e as u32).collect();

        // Flatten word_embeddings to [batch * num_words, hidden_dim]
        let word_embeddings_concrete = word_embeddings.to_concrete();
        let flat_embeddings = word_embeddings_concrete
            .reshape([batch_size * num_words, hidden_dim])
            .to_concrete();

        // Build offset indices for batch processing
        let mut start_offset_indices: Vec<u32> = Vec::with_capacity(batch_size * num_spans);
        let mut end_offset_indices: Vec<u32> = Vec::with_capacity(batch_size * num_spans);

        for batch_idx in 0..batch_size {
            let offset = (batch_idx * num_words) as u32;
            for &start in &start_indices {
                start_offset_indices.push(start + offset);
            }
        }
        for batch_idx in 0..batch_size {
            let offset = (batch_idx * num_words) as u32;
            for &end in &end_indices {
                end_offset_indices.push(end + offset);
            }
        }

        let start_idx_tensor = Tensor::new(device, &start_offset_indices);
        let end_idx_tensor = Tensor::new(device, &end_offset_indices);

        // Gather start and end embeddings
        let start_emb = flat_embeddings.index_select(0, &start_idx_tensor);
        let end_emb = flat_embeddings.index_select(0, &end_idx_tensor);

        // Reshape to [batch, num_spans, hidden_dim]
        let start_emb = start_emb
            .reshape([batch_size, num_spans, hidden_dim])
            .to_concrete();
        let end_emb = end_emb
            .reshape([batch_size, num_spans, hidden_dim])
            .to_concrete();

        (start_emb, end_emb)
    }
}
