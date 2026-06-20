//! # rgliner
//!
//! GLiNER bi-encoder Named Entity Recognition for Rust.
//!
//! GLiNER (Generalist Lightweight Model for Named Entity Recognition) identifies
//! arbitrary entity types at inference time using natural language labels.
//!
//! ## Usage
//!
//! ```rust, no_run
//! use rgliner::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut gliner = Gliner::new().await?;
//!
//!     let labels = ["person", "organization", "location"];
//!     let text = "Apple Inc. was founded by Steve Jobs in California.";
//!
//!     let entities = gliner.extract(text, &labels).await?;
//!     for entity in entities {
//!         println!("{}: {} ({:.2})", entity.label, entity.text, entity.score);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Label Caching
//!
//! For production workloads with fixed label sets, you can pre-compute label
//! embeddings for significant speedup:
//!
//! ```rust, no_run
//! use rgliner::*;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let mut gliner = Gliner::new().await?;
//!
//! // Pre-compute label embeddings once
//! let labels = ["person", "organization", "location"];
//! gliner.cache_labels(&labels).await?;
//!
//! // Fast inference with cached labels
//! let documents = ["Apple Inc. was founded by Steve Jobs.", "Microsoft is in Seattle."];
//! for text in documents {
//!     let entities = gliner.extract_with_cached_labels(text).await?;
//!     // Process entities...
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Relation Extraction (GLiNER-RelEx)
//!
//! For joint NER and relation extraction, use the `relex` module:
//!
//! ```rust, no_run
//! use rgliner::relex::*;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let relex = GlinerRelEx::builder()
//!     .with_source(GlinerRelExSource::relex_multi())
//!     .build()
//!     .await?;
//!
//! let (entities, relations) = relex.extract(
//!     "Apple was founded by Steve Jobs.",
//!     &["person", "organization"],
//!     &["founded by"],
//! ).await?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

mod config;
mod decoding;
mod error;
mod raw;
pub mod relation_decoding;
pub mod relex;
pub mod relex_tokenization;
mod source;
mod tokenization;

pub use config::GlinerConfig;
pub use decoding::{Decoder, DecodingMode, Entity};
pub use error::{GlinerError, GlinerLoadingError};
pub use rbert::raw::{ModernBertConfig, ModernBertModel};
pub use source::GlinerSource;

/// Deduplicate entities appearing in more than one overlapping chunk, keeping the
/// highest-scoring occurrence and sorting by span position.
fn merge_entities(entities: &mut Vec<Entity>) {
    entities.sort_by(|a, b| {
        a.start_char
            .cmp(&b.start_char)
            .then_with(|| a.end_char.cmp(&b.end_char))
            .then_with(|| a.label.cmp(&b.label))
    });
    entities.dedup_by(|b, a| {
        if a.start_char == b.start_char && a.end_char == b.end_char && a.label == b.label {
            if b.score > a.score {
                a.score = b.score;
            }
            true
        } else {
            false
        }
    });
}


use fusor::{Device, Tensor, VarBuilder};
use kalosm_common::Cache;
use kalosm_model_types::{FileSource, ModelLoadingProgress};
use rbert::BertSource;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::raw::{CachedLabels, LabelEncoder, SpanLayer, TextEncoder};
use crate::tokenization::{first_subtoken_pooling, TokenizedText, WordTokenizer};

async fn default_device() -> Device {
    Device::gpu().await.unwrap_or_else(|_| Device::cpu())
}

/// Download a model artifact from `source` through `cache`, reporting progress
/// under the label `"<label> (<source>)"`. Shared by both model loaders.
pub(crate) async fn download_bytes(
    cache: &Cache,
    source: &FileSource,
    label: &str,
    progress_handler: &mut impl FnMut(ModelLoadingProgress),
) -> Result<Vec<u8>, GlinerLoadingError> {
    let mut create_progress =
        ModelLoadingProgress::downloading_progress(format!("{label} ({source})"));
    Ok(cache
        .get_bytes(source, |progress| progress_handler(create_progress(progress)))
        .await?)
}

/// Builder for constructing a [`Gliner`] model.
#[derive(Default)]
pub struct GlinerBuilder {
    source: GlinerSource,
    cache: Cache,
    device: Option<Device>,
    decoding_mode: DecodingMode,
    threshold: f32,
    max_width: Option<usize>,
}

impl GlinerBuilder {
    /// Set the model source.
    pub fn with_source(mut self, source: GlinerSource) -> Self {
        self.source = source;
        self
    }

    /// Set the decoding mode (Flat or Nested).
    pub fn with_decoding_mode(mut self, mode: DecodingMode) -> Self {
        self.decoding_mode = mode;
        self
    }

    /// Set the confidence threshold (default 0.5).
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the maximum span width (overrides config).
    pub fn with_max_width(mut self, max_width: usize) -> Self {
        self.max_width = Some(max_width);
        self
    }

    /// Set the device.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the cache location.
    pub fn with_cache(mut self, cache: Cache) -> Self {
        self.cache = cache;
        self
    }

    /// Build the model.
    pub async fn build(self) -> Result<Gliner, GlinerLoadingError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }

    /// Build the model with a loading handler.
    pub async fn build_with_loading_handler(
        self,
        loading_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> Result<Gliner, GlinerLoadingError> {
        Gliner::from_builder(self, loading_handler).await
    }
}

/// GLiNER Named Entity Recognition model.
///
/// The bi-encoder architecture enables efficient NER with arbitrary entity types.
/// Labels are encoded independently from text, allowing pre-computation and caching.
pub struct Gliner {
    text_encoder: TextEncoder,
    label_encoder: LabelEncoder,
    span_layer: SpanLayer,
    tokenizer: Arc<WordTokenizer>,
    decoder: Decoder,
    device: Device,
    max_width: usize,
    /// Cached label embeddings for repeated inference.
    cached_labels: Option<CachedLabels>,
}

impl Gliner {
    /// Create a new builder.
    pub fn builder() -> GlinerBuilder {
        GlinerBuilder {
            threshold: 0.5,
            ..Default::default()
        }
    }

    /// Create with default settings (base model).
    pub async fn new() -> Result<Self, GlinerLoadingError> {
        Self::builder().build().await
    }

    async fn from_builder(
        builder: GlinerBuilder,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> Result<Self, GlinerLoadingError> {
        let GlinerBuilder {
            source,
            cache,
            device,
            decoding_mode,
            threshold,
            max_width: max_width_override,
        } = builder;

        // Download config file
        let config_bytes =
            download_bytes(&cache, &source.config, "Config", &mut progress_handler).await?;
        let config =
            GlinerConfig::from_json(&config_bytes).map_err(GlinerLoadingError::LoadConfig)?;

        // Download tokenizer
        let tokenizer_bytes =
            download_bytes(&cache, &source.tokenizer, "Tokenizer", &mut progress_handler).await?;
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer_bytes).map_err(GlinerLoadingError::LoadTokenizer)?;
        let word_tokenizer = WordTokenizer::new(tokenizer, config.should_add_special_tokens());

        // Download main model weights, then the label encoder (warms the cache;
        // the label encoder is reloaded from cache via BertSource below).
        let model_bytes =
            download_bytes(&cache, &source.model, "Text Encoder", &mut progress_handler).await?;
        let _label_bytes =
            download_bytes(&cache, &source.label_encoder, "Label Encoder", &mut progress_handler)
                .await?;

        // Initialize device
        let device = match device {
            Some(device) => device,
            None => default_device().await,
        };

        // Load text encoder
        let mut model_cursor = std::io::Cursor::new(&model_bytes);
        let mut text_vb = VarBuilder::from_gguf(&mut model_cursor)
            .map_err(|err| GlinerLoadingError::LoadModel(fusor::Error::from(err)))?;

        let text_encoder = TextEncoder::load(&device, &mut text_vb)?;

        // Load span layer from main model weights
        let max_width = max_width_override.unwrap_or(config.max_width);
        let span_layer = SpanLayer::load(&device, &mut text_vb)?;

        // Load label encoder
        let label_encoder_source = BertSource::new()
            .with_model(source.label_encoder.clone())
            .with_config(source.label_encoder_config.clone())
            .with_tokenizer(source.label_encoder_tokenizer.clone());

        // Create projection VarBuilder from main model
        let mut model_cursor2 = std::io::Cursor::new(&model_bytes);
        let mut proj_vb = VarBuilder::from_gguf(&mut model_cursor2)
            .map_err(|err| GlinerLoadingError::LoadModel(fusor::Error::from(err)))?;

        let label_encoder = LabelEncoder::load(&device, &mut proj_vb, label_encoder_source).await?;

        let decoder = Decoder::new(threshold, decoding_mode);

        Ok(Self {
            text_encoder,
            label_encoder,
            span_layer,
            tokenizer: Arc::new(word_tokenizer),
            decoder,
            device,
            max_width,
            cached_labels: None,
        })
    }

    /// Cache label embeddings for repeated inference with the same labels.
    ///
    /// This significantly speeds up inference when using fixed label sets.
    pub async fn cache_labels(&mut self, labels: &[&str]) -> Result<(), GlinerError> {
        // `materialized()` severs the lazy encoder graph into a standalone
        // buffer; `to_concrete()` would only clone the lazy GPU tensor and
        // re-run the encoder on every reuse.
        let label_embeddings = self
            .label_encoder
            .encode_labels(labels)
            .await?;
        self.cached_labels = Some(CachedLabels::new(
            labels.iter().map(|s| s.to_string()).collect(),
            label_embeddings,
        ));
        Ok(())
    }

    /// Clear cached label embeddings.
    pub fn clear_label_cache(&mut self) {
        self.cached_labels = None;
    }

    /// Check if labels are cached.
    pub fn has_cached_labels(&self) -> bool {
        self.cached_labels.is_some()
    }

    /// Extract named entities from text.
    pub async fn extract(
        &mut self,
        text: &str,
        labels: &[&str],
    ) -> Result<Vec<Entity>, GlinerError> {
        let mut results = self.extract_batch(&[text], labels).await?;
        Ok(results.pop().unwrap_or_default())
    }

    /// Extract named entities from text, chunking the input first so long documents
    /// that would otherwise be truncated by the text encoder's context window still
    /// get full coverage.
    ///
    /// Uses the model's own tokenizer to pack whole words into chunks of at most
    /// `token_budget` subtokens, with roughly 15% token overlap between adjacent
    /// chunks. Each chunk is scored independently; entity offsets are remapped back
    /// into the original text and deduped across overlapping windows (keeping the
    /// highest score per span+label).
    ///
    /// `token_budget` defaults to 128 — empirically the sweet spot for the edge
    /// variant's span-scoring quality. Larger budgets approach the context limit
    /// but hurt F1; much smaller budgets hurt recall.
    pub async fn extract_auto(
        &mut self,
        text: &str,
        labels: &[&str],
        token_budget: Option<usize>,
    ) -> Result<Vec<Entity>, GlinerError> {
        let budget = token_budget.unwrap_or(128);
        let ranges = crate::tokenization::token_packed_ranges(
            &self.tokenizer.tokenizer,
            text,
            budget,
            budget / 7,
        )?;
        if ranges.len() <= 1 {
            return self.extract(text, labels).await;
        }

        let chunk_texts: Vec<&str> = ranges.iter().map(|r| &text[r.clone()]).collect();
        let per_chunk = self.extract_batch(&chunk_texts, labels).await?;

        let mut all: Vec<Entity> = Vec::new();
        for (range, entities) in ranges.iter().zip(per_chunk) {
            let offset = range.start;
            for mut ent in entities {
                ent.start_char += offset;
                ent.end_char += offset;
                all.push(ent);
            }
        }
        merge_entities(&mut all);
        Ok(all)
    }

    /// Extract named entities using cached labels.
    ///
    /// Panics if no labels are cached.
    pub async fn extract_with_cached_labels(
        &mut self,
        text: &str,
    ) -> Result<Vec<Entity>, GlinerError> {
        let labels: Vec<String> = self
            .cached_labels
            .as_ref()
            .expect("No labels cached. Call cache_labels first.")
            .labels
            .clone();
        let labels: Vec<&str> = labels.iter().map(|label| label.as_str()).collect();
        self.extract(text, &labels).await
    }

    /// Extract named entities from a batch of texts.
    pub async fn extract_batch(
        &mut self,
        texts: &[&str],
        labels: &[&str],
    ) -> Result<Vec<Vec<Entity>>, GlinerError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Get label embeddings, reusing the cache when the label set is
        // unchanged. The label encoder is independent of the input text, so a
        // fixed label set (the common case when extracting over many texts)
        // only needs encoding once. Auto-populate the cache on a miss so the
        // default `extract`/`extract_batch` path amortizes label encoding
        // without requiring an explicit `cache_labels` call.
        let labels_match = self.cached_labels.as_ref().is_some_and(|cached| {
            cached.labels.len() == labels.len()
                && cached
                    .labels
                    .iter()
                    .zip(labels.iter())
                    .all(|(cached, label)| cached == label)
        });
        let label_embeddings = if labels_match {
            self.cached_labels.as_ref().unwrap().embeddings.clone()
        } else {
            // `materialized()` (not `to_concrete()`) resolves the encoder graph
            // into a standalone output buffer, severing the lazy graph so reuse
            // does not re-run the label encoder. `to_concrete()` only clones the
            // lazy GPU tensor, which would drag the whole encoder into every
            // subsequent extract.
            let embeddings = self.label_encoder.encode_labels(labels).await?;
            self.cached_labels = Some(CachedLabels::new(
                labels.iter().map(|s| s.to_string()).collect(),
                embeddings.clone(),
            ));
            embeddings
        };

        self.extract_internal_batch(texts, labels, &label_embeddings)
            .await
    }

    async fn extract_internal_batch(
        &self,
        texts: &[&str],
        labels: &[&str],
        label_embeddings: &Tensor<2, f32>,
    ) -> Result<Vec<Vec<Entity>>, GlinerError> {
        let tokenized = self.tokenizer.tokenize_batch(texts)?;
        if tokenized.iter().all(|tokenized| tokenized.num_words == 0) {
            return Ok(vec![Vec::new(); texts.len()]);
        }

        let (token_ids, attention_mask) = self.build_batched_inputs(&tokenized);

        let token_embeddings = self.text_encoder.forward(&token_ids, Some(&attention_mask));

        // Python's bi-encoder span model pools transformer token embeddings
        // directly to words; the checkpoint still contains LSTM weights, but
        // that path is not used in BaseBiEncoderModel.get_representations().
        let (word_embeddings, _word_mask) =
            first_subtoken_pooling(&token_embeddings, &tokenized, &self.device);

        let spans_per_batch: Vec<Vec<(usize, usize)>> = tokenized
            .iter()
            .map(|tokenized| self.enumerate_spans(tokenized.num_words))
            .collect();
        let span_counts: Vec<usize> = spans_per_batch.iter().map(Vec::len).collect();
        let total_spans: usize = span_counts.iter().sum();

        if total_spans == 0 {
            return Ok(vec![Vec::new(); texts.len()]);
        }

        let (flat_span_embeddings, _) =
            self.span_layer
                .forward_for_spans_batched(&word_embeddings, &spans_per_batch, &self.device);

        let labels_t = label_embeddings.t();
        let flat_scores = flat_span_embeddings.mat_mul(&labels_t);
        let tensor_slice = flat_scores.as_slice().await?;
        let scores_data: Vec<f32> = tensor_slice
            .as_slice()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp())) // sigmoid
            .collect();

        let num_labels = label_embeddings.shape()[0];
        let mut results = Vec::with_capacity(texts.len());
        let mut score_offset = 0usize;

        for (batch_idx, tokenized) in tokenized.iter().enumerate() {
            let span_count = span_counts[batch_idx];
            if span_count == 0 {
                results.push(Vec::new());
                continue;
            }

            let next_offset = score_offset + span_count * num_labels;
            let entities = self.decoder.decode(
                &scores_data[score_offset..next_offset],
                span_count,
                num_labels,
                &spans_per_batch[batch_idx],
                &tokenized.word_offsets,
                labels,
                texts[batch_idx],
            );
            results.push(entities);
            score_offset = next_offset;
        }

        Ok(results)
    }

    fn build_batched_inputs(&self, tokenized: &[TokenizedText]) -> (Tensor<2, u32>, Tensor<2, u32>) {
        let batch_size = tokenized.len();
        let max_seq_len = tokenized
            .iter()
            .map(|tokenized| tokenized.token_ids.len())
            .max()
            .unwrap_or(1);
        let pad_id = self.tokenizer.pad_id();

        let mut token_ids = vec![pad_id; batch_size * max_seq_len];
        let mut attention_mask = vec![0u32; batch_size * max_seq_len];

        for (batch_idx, item) in tokenized.iter().enumerate() {
            let offset = batch_idx * max_seq_len;
            let len = item.token_ids.len();
            token_ids[offset..offset + len].copy_from_slice(&item.token_ids);
            attention_mask[offset..offset + len].copy_from_slice(&item.attention_mask);
        }

        (
            Tensor::new(&self.device, &token_ids)
                .reshape([batch_size, max_seq_len])
                .to_concrete(),
            Tensor::new(&self.device, &attention_mask)
                .reshape([batch_size, max_seq_len])
                .to_concrete(),
        )
    }

    fn enumerate_spans(&self, num_words: usize) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        for start in 0..num_words {
            let max_width = self.max_width.min(num_words - start);
            for width in 1..=max_width {
                spans.push((start, start + width - 1));
            }
        }
        spans
    }

    /// Get the maximum span width.
    pub fn max_width(&self) -> usize {
        self.max_width
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}
