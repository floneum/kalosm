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
//! for text in documents {
//!     let entities = gliner.extract_with_cached_labels(&text).await?;
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
use kalosm_model_types::ModelLoadingProgress;
use rbert::BertSource;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::raw::{CachedLabels, LabelEncoder, Scorer, SpanLayer, TextEncoder};
use crate::tokenization::{first_subtoken_pooling, WordTokenizer};

async fn default_device() -> Device {
    Device::gpu().await.unwrap_or_else(|_| Device::cpu())
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
    #[cfg(feature = "tokio")]
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
        let config_source = format!("Config ({})", source.config);
        let mut create_progress = ModelLoadingProgress::downloading_progress(config_source);
        let config_bytes = cache
            .get_bytes(&source.config, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        let config =
            GlinerConfig::from_json(&config_bytes).map_err(GlinerLoadingError::LoadConfig)?;

        // Download tokenizer
        let tokenizer_source = format!("Tokenizer ({})", source.tokenizer);
        let mut create_progress = ModelLoadingProgress::downloading_progress(tokenizer_source);
        let tokenizer_bytes = cache
            .get_bytes(&source.tokenizer, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        let tokenizer =
            Tokenizer::from_bytes(&tokenizer_bytes).map_err(GlinerLoadingError::LoadTokenizer)?;
        let word_tokenizer = WordTokenizer::new(tokenizer, config.should_add_special_tokens());

        // Download main model weights
        let model_source = format!("Text Encoder ({})", source.model);
        let mut create_progress = ModelLoadingProgress::downloading_progress(model_source);
        let model_bytes = cache
            .get_bytes(&source.model, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        // Download label encoder weights
        let label_source = format!("Label Encoder ({})", source.label_encoder);
        let mut create_progress = ModelLoadingProgress::downloading_progress(label_source);
        let _label_bytes = cache
            .get_bytes(&source.label_encoder, |progress| {
                progress_handler(create_progress(progress))
            })
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
        let span_layer = SpanLayer::load(&device, &mut text_vb, max_width)?;

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
        let label_embeddings = self
            .label_encoder
            .encode_labels(labels)
            .await?
            .to_concrete();
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

        // Get label embeddings (compute if not cached or labels differ)
        let label_embeddings = if let Some(ref cached) = self.cached_labels {
            let cached_labels: Vec<&str> = cached.labels.iter().map(|s| s.as_str()).collect();
            if cached_labels == labels {
                cached.embeddings.clone()
            } else {
                self.label_encoder.encode_labels(labels).await?
            }
        } else {
            self.label_encoder.encode_labels(labels).await?
        };

        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let entities = self
                .extract_internal(text, labels, &label_embeddings)
                .await?;
            results.push(entities);
        }

        Ok(results)
    }

    async fn extract_internal(
        &self,
        text: &str,
        labels: &[&str],
        label_embeddings: &Tensor<2, f32>,
    ) -> Result<Vec<Entity>, GlinerError> {
        // 1. Tokenize text
        let tokenized = self.tokenizer.tokenize(text)?;

        if tokenized.num_words == 0 {
            return Ok(Vec::new());
        }

        // 2. Prepare input tensors
        let token_ids = Tensor::new(&self.device, &tokenized.token_ids);
        let token_ids: Tensor<2, u32> = token_ids.unsqueeze(0).to_concrete();

        let attention_mask = Tensor::new(&self.device, &tokenized.attention_mask);
        let attention_mask: Tensor<2, u32> = attention_mask.unsqueeze(0).to_concrete();

        // 3. Encode text
        let token_embeddings = self.text_encoder.forward(&token_ids, Some(&attention_mask));

        // Python's bi-encoder span model pools transformer token embeddings
        // directly to words; the checkpoint still contains LSTM weights, but
        // that path is not used in BaseBiEncoderModel.get_representations().
        let (word_embeddings, _word_mask) =
            first_subtoken_pooling(&token_embeddings, &[tokenized.clone()], &self.device);

        // 4. Generate span representations
        let (span_embeddings, span_indices) =
            self.span_layer.forward(&word_embeddings, &self.device);

        // 5. Score spans against labels
        let scores = Scorer::forward(&span_embeddings, label_embeddings);

        // 6. Decode predictions
        let shape = scores.shape();
        let num_spans = shape[1];
        let num_labels = shape[2];

        // Get scores for first batch item and apply sigmoid
        let flat_scores: Tensor<2, f32> = scores.squeeze(0).to_concrete();
        let tensor_slice = flat_scores.as_slice().await?;
        let scores_data: Vec<f32> = tensor_slice
            .as_slice()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp())) // sigmoid
            .collect();

        let entities = self.decoder.decode(
            &scores_data,
            num_spans,
            num_labels,
            &span_indices,
            &tokenized.word_offsets,
            labels,
            text,
        );

        Ok(entities)
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

#[cfg(test)]
mod gpu_parity_tests {
    use super::*;
    use fusor::layers::{Embedding, LayerNorm};
    use std::path::Path;

    fn local_edge_source() -> Option<GlinerSource> {
        let weights_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
        let model_path = weights_dir.join("gliner-edge.gguf");
        let label_encoder_path = weights_dir.join("gliner-edge-label-encoder.gguf");
        if model_path.exists() && label_encoder_path.exists() {
            Some(GlinerSource::local(model_path, label_encoder_path))
        } else {
            None
        }
    }

    async fn load_local_edge(device: Device) -> Result<Gliner, GlinerLoadingError> {
        Gliner::builder()
            .with_source(local_edge_source().expect("local edge checkpoint missing"))
            .with_device(device)
            .build()
            .await
    }

    async fn tensor_values<const R: usize>(tensor: &Tensor<R, f32>) -> fusor::Result<Vec<f32>> {
        let slice = tensor.clone().as_slice().await?;
        Ok(slice.as_slice().iter().copied().collect())
    }

    fn load_qmatrix_tensor_from_gguf(
        gguf_bytes: &[u8],
        scope: &[&str],
        tensor_name: &str,
        device: &Device,
    ) -> fusor::QMatrix {
        let mut cursor = std::io::Cursor::new(gguf_bytes);
        let mut vb = VarBuilder::from_gguf(&mut cursor).unwrap();
        let mut vb = if scope.is_empty() {
            vb
        } else {
            vb.pp(scope.join("."))
        };
        vb.get(tensor_name, device).unwrap()
    }

    fn load_qmatrix_from_gguf(
        gguf_bytes: &[u8],
        scope: &[&str],
        device: &Device,
    ) -> fusor::QMatrix {
        load_qmatrix_tensor_from_gguf(gguf_bytes, scope, "weight", device)
    }

    fn load_layer_norm_from_gguf(
        gguf_bytes: &[u8],
        scope: &[&str],
        device: &Device,
        eps: f32,
    ) -> LayerNorm<1, f32> {
        let mut cursor = std::io::Cursor::new(gguf_bytes);
        let mut vb = VarBuilder::from_gguf(&mut cursor).unwrap();
        let scope = if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        };
        if let Some(scope) = scope {
            let mut scoped = vb.pp(scope);
            LayerNorm::load(device, &mut scoped, eps).unwrap()
        } else {
            LayerNorm::load(device, &mut vb, eps).unwrap()
        }
    }

    fn load_modern_bert_config_from_gguf(gguf_bytes: &[u8]) -> ModernBertConfig {
        let mut cursor = std::io::Cursor::new(gguf_bytes);
        let mut vb = VarBuilder::from_gguf(&mut cursor).unwrap();
        ModernBertConfig::from_gguf(&mut vb.pp("text")).unwrap()
    }

    async fn print_diff<const R: usize>(
        name: &str,
        cpu: &Tensor<R, f32>,
        gpu: &Tensor<R, f32>,
    ) -> fusor::Result<f32> {
        assert_eq!(cpu.shape(), gpu.shape(), "{name} shape mismatch");

        let cpu_values = tensor_values(cpu).await?;
        let gpu_values = tensor_values(gpu).await?;

        let mut max_abs_diff = 0.0f32;
        let mut mean_abs_diff = 0.0f32;

        for (cpu, gpu) in cpu_values.iter().zip(&gpu_values) {
            let diff = (cpu - gpu).abs();
            max_abs_diff = max_abs_diff.max(diff);
            mean_abs_diff += diff;
        }

        mean_abs_diff /= cpu_values.len().max(1) as f32;

        println!(
            "{name}: shape={:?}, max_abs_diff={max_abs_diff:.6}, mean_abs_diff={mean_abs_diff:.6}",
            cpu.shape()
        );

        Ok(max_abs_diff)
    }

    fn build_text_inputs(
        tokenized: &crate::tokenization::TokenizedText,
        device: &Device,
    ) -> (Tensor<2, u32>, Tensor<2, u32>) {
        let token_ids = Tensor::new(device, &tokenized.token_ids);
        let token_ids: Tensor<2, u32> = token_ids.unsqueeze(0).to_concrete();

        let attention_mask = Tensor::new(device, &tokenized.attention_mask);
        let attention_mask: Tensor<2, u32> = attention_mask.unsqueeze(0).to_concrete();

        (token_ids, attention_mask)
    }

    #[tokio::test]
    #[ignore = "requires local edge checkpoint files and a working GPU device"]
    async fn debug_cpu_gpu_parity_for_local_edge_checkpoint() {
        if local_edge_source().is_none() {
            eprintln!("Skipping GPU parity test: local edge checkpoint files are missing.");
            return;
        }

        let gpu_device = match std::panic::catch_unwind(Device::gpu_blocking) {
            Ok(Ok(device)) => device,
            Ok(Err(err)) => {
                eprintln!("Skipping GPU parity test: failed to create GPU device: {err}");
                return;
            }
            Err(_) => {
                eprintln!("Skipping GPU parity test: GPU device creation panicked.");
                return;
            }
        };

        let cpu_device = Device::cpu();
        let weights_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");
        let model_bytes = std::fs::read(weights_dir.join("gliner-edge.gguf")).unwrap();
        let label_encoder_bytes =
            std::fs::read(weights_dir.join("gliner-edge-label-encoder.gguf")).unwrap();

        let mut cpu = load_local_edge(cpu_device.clone()).await.unwrap();
        let mut gpu = load_local_edge(gpu_device.clone()).await.unwrap();

        let labels = ["person", "organization", "location"];
        let text = "Google was founded in Mountain View.";

        let cpu_tokenized = cpu.tokenizer.tokenize(text).unwrap();
        let gpu_tokenized = gpu.tokenizer.tokenize(text).unwrap();
        assert_eq!(cpu_tokenized.token_ids, gpu_tokenized.token_ids);
        assert_eq!(cpu_tokenized.attention_mask, gpu_tokenized.attention_mask);
        assert_eq!(cpu_tokenized.word_offsets, gpu_tokenized.word_offsets);

        let cpu_text_embedding = Embedding::new(load_qmatrix_from_gguf(
            &model_bytes,
            &["text", "token_embd"],
            &cpu_device,
        ));
        let gpu_text_embedding = Embedding::new(load_qmatrix_from_gguf(
            &model_bytes,
            &["text", "token_embd"],
            &gpu_device,
        ));
        let cpu_text_ids = Tensor::new(&cpu_device, &cpu_tokenized.token_ids);
        let gpu_text_ids = Tensor::new(&gpu_device, &gpu_tokenized.token_ids);
        let cpu_text_embedding_lookup: Tensor<2, f32> = cpu_text_embedding.forward(&cpu_text_ids);
        let gpu_text_embedding_lookup: Tensor<2, f32> = gpu_text_embedding.forward(&gpu_text_ids);
        let _ = print_diff(
            "raw_text_token_embedding_lookup",
            &cpu_text_embedding_lookup,
            &gpu_text_embedding_lookup,
        )
        .await
        .unwrap();

        let cpu_text_embd_norm =
            load_layer_norm_from_gguf(&model_bytes, &["text", "embd_norm"], &cpu_device, 1e-6);
        let gpu_text_embd_norm =
            load_layer_norm_from_gguf(&model_bytes, &["text", "embd_norm"], &gpu_device, 1e-6);
        let cpu_text_embedding_lookup_3d: Tensor<3, f32> =
            cpu_text_embedding_lookup.unsqueeze(0).to_concrete();
        let gpu_text_embedding_lookup_3d: Tensor<3, f32> =
            gpu_text_embedding_lookup.unsqueeze(0).to_concrete();
        let cpu_text_after_embd_norm = cpu_text_embd_norm.forward(&cpu_text_embedding_lookup_3d);
        let gpu_text_after_embd_norm = gpu_text_embd_norm.forward(&gpu_text_embedding_lookup_3d);
        let _ = print_diff(
            "text_after_embd_norm",
            &cpu_text_after_embd_norm,
            &gpu_text_after_embd_norm,
        )
        .await
        .unwrap();

        let cpu_layer0_qkv = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.0"],
            "attn_qkv.weight",
            &cpu_device,
        );
        let gpu_layer0_qkv = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.0"],
            "attn_qkv.weight",
            &gpu_device,
        );
        let cpu_layer0_qkv_projection = cpu_text_after_embd_norm.q_mat_mul(&cpu_layer0_qkv);
        let gpu_layer0_qkv_projection = gpu_text_after_embd_norm.q_mat_mul(&gpu_layer0_qkv);
        let _ = print_diff(
            "text_layer0_qkv_projection",
            &cpu_layer0_qkv_projection,
            &gpu_layer0_qkv_projection,
        )
        .await
        .unwrap();

        let label_probe_indices = [0u32, 1, 2, 17, 101, 257, 1024];
        let cpu_label_token_embedding = Embedding::new(load_qmatrix_from_gguf(
            &label_encoder_bytes,
            &["token_embd"],
            &cpu_device,
        ));
        let gpu_label_token_embedding = Embedding::new(load_qmatrix_from_gguf(
            &label_encoder_bytes,
            &["token_embd"],
            &gpu_device,
        ));
        let cpu_label_ids = Tensor::new(&cpu_device, &label_probe_indices);
        let gpu_label_ids = Tensor::new(&gpu_device, &label_probe_indices);
        let cpu_label_embedding_lookup: Tensor<2, f32> =
            cpu_label_token_embedding.forward(&cpu_label_ids);
        let gpu_label_embedding_lookup: Tensor<2, f32> =
            gpu_label_token_embedding.forward(&gpu_label_ids);
        let _ = print_diff(
            "raw_label_token_embedding_lookup",
            &cpu_label_embedding_lookup,
            &gpu_label_embedding_lookup,
        )
        .await
        .unwrap();

        let single_label = ["organization"];
        let cpu_single_label_embeddings = cpu
            .label_encoder
            .encode_labels(&single_label)
            .await
            .unwrap();
        let gpu_single_label_embeddings = gpu
            .label_encoder
            .encode_labels(&single_label)
            .await
            .unwrap();
        let _ = print_diff(
            "single_label_embeddings",
            &cpu_single_label_embeddings,
            &gpu_single_label_embeddings,
        )
        .await
        .unwrap();

        let (cpu_label_token_embeddings, cpu_label_attention_mask) = cpu
            .label_encoder
            .debug_sentence_token_embeddings_and_mask(&labels)
            .unwrap();
        let (gpu_label_token_embeddings, gpu_label_attention_mask) = gpu
            .label_encoder
            .debug_sentence_token_embeddings_and_mask(&labels)
            .unwrap();
        let _ = print_diff(
            "label_token_embeddings",
            &cpu_label_token_embeddings,
            &gpu_label_token_embeddings,
        )
        .await
        .unwrap();
        let _ = print_diff(
            "label_attention_mask",
            &cpu_label_attention_mask.cast(),
            &gpu_label_attention_mask.cast(),
        )
        .await
        .unwrap();

        let (cpu_label_states, _) = cpu
            .label_encoder
            .debug_sentence_hidden_states(&labels)
            .unwrap();
        let (gpu_label_states, _) = gpu
            .label_encoder
            .debug_sentence_hidden_states(&labels)
            .unwrap();
        for (idx, (cpu_state, gpu_state)) in
            cpu_label_states.iter().zip(&gpu_label_states).enumerate()
        {
            let name = if idx == 0 {
                "label_post_embeddings".to_string()
            } else {
                format!("label_post_layer_{}", idx - 1)
            };
            let _ = print_diff(&name, cpu_state, gpu_state).await.unwrap();
        }

        let (cpu_label_layer0_attention, cpu_label_layer0_intermediate, cpu_label_layer0_output) =
            cpu.label_encoder
                .debug_sentence_first_layer(&labels)
                .unwrap();
        let (gpu_label_layer0_attention, gpu_label_layer0_intermediate, gpu_label_layer0_output) =
            gpu.label_encoder
                .debug_sentence_first_layer(&labels)
                .unwrap();
        let _ = print_diff(
            "label_layer0_attention_output",
            &cpu_label_layer0_attention,
            &gpu_label_layer0_attention,
        )
        .await
        .unwrap();
        let (
            cpu_label_layer0_query,
            cpu_label_layer0_key,
            cpu_label_layer0_value,
            cpu_label_layer0_self_output,
            cpu_label_layer0_attention_output_debug,
        ) = cpu
            .label_encoder
            .debug_sentence_first_layer_attention(&labels)
            .unwrap();
        let (
            gpu_label_layer0_query,
            gpu_label_layer0_key,
            gpu_label_layer0_value,
            gpu_label_layer0_self_output,
            gpu_label_layer0_attention_output_debug,
        ) = gpu
            .label_encoder
            .debug_sentence_first_layer_attention(&labels)
            .unwrap();
        let _ = print_diff(
            "label_layer0_query",
            &cpu_label_layer0_query,
            &gpu_label_layer0_query,
        )
        .await
        .unwrap();
        let _ = print_diff(
            "label_layer0_key",
            &cpu_label_layer0_key,
            &gpu_label_layer0_key,
        )
        .await
        .unwrap();
        let _ = print_diff(
            "label_layer0_value",
            &cpu_label_layer0_value,
            &gpu_label_layer0_value,
        )
        .await
        .unwrap();
        let _ = print_diff(
            "label_layer0_self_attention_output",
            &cpu_label_layer0_self_output,
            &gpu_label_layer0_self_output,
        )
        .await
        .unwrap();
        let _ = print_diff(
            "label_layer0_attention_output_debug_split",
            &cpu_label_layer0_attention_output_debug,
            &gpu_label_layer0_attention_output_debug,
        )
        .await
        .unwrap();
        let _ = print_diff(
            "label_layer0_intermediate_output",
            &cpu_label_layer0_intermediate,
            &gpu_label_layer0_intermediate,
        )
        .await
        .unwrap();
        let _ = print_diff(
            "label_layer0_output_debug",
            &cpu_label_layer0_output,
            &gpu_label_layer0_output,
        )
        .await
        .unwrap();

        let cpu_label_mean_pool = cpu.label_encoder.debug_sentence_mean_pool(&labels).unwrap();
        let gpu_label_mean_pool = gpu.label_encoder.debug_sentence_mean_pool(&labels).unwrap();
        let _ = print_diff(
            "label_mean_pool",
            &cpu_label_mean_pool,
            &gpu_label_mean_pool,
        )
        .await
        .unwrap();

        let cpu_label_sentence_embeddings = cpu
            .label_encoder
            .debug_sentence_embeddings(&labels)
            .await
            .unwrap();
        let gpu_label_sentence_embeddings = gpu
            .label_encoder
            .debug_sentence_embeddings(&labels)
            .await
            .unwrap();
        let _ = print_diff(
            "label_sentence_embeddings",
            &cpu_label_sentence_embeddings,
            &gpu_label_sentence_embeddings,
        )
        .await
        .unwrap();

        let cpu_projected_label_embeddings = cpu
            .label_encoder
            .debug_projection(&cpu_label_sentence_embeddings);
        let gpu_projected_label_embeddings = gpu
            .label_encoder
            .debug_projection(&gpu_label_sentence_embeddings);
        let _ = print_diff(
            "label_projected_embeddings",
            &cpu_projected_label_embeddings,
            &gpu_projected_label_embeddings,
        )
        .await
        .unwrap();

        let cpu_label_embeddings = cpu.label_encoder.encode_labels(&labels).await.unwrap();
        let gpu_label_embeddings = gpu.label_encoder.encode_labels(&labels).await.unwrap();
        let _ = print_diff(
            "label_embeddings",
            &cpu_label_embeddings,
            &gpu_label_embeddings,
        )
        .await
        .unwrap();

        let (cpu_token_ids, cpu_attention_mask) = build_text_inputs(&cpu_tokenized, &cpu_device);
        let (gpu_token_ids, gpu_attention_mask) = build_text_inputs(&gpu_tokenized, &gpu_device);

        let cpu_token_embeddings = cpu
            .text_encoder
            .forward(&cpu_token_ids, Some(&cpu_attention_mask));
        let gpu_token_embeddings = gpu
            .text_encoder
            .forward(&gpu_token_ids, Some(&gpu_attention_mask));
        let cpu_text_states = cpu
            .text_encoder
            .debug_hidden_states(&cpu_token_ids, Some(&cpu_attention_mask));
        let gpu_text_states = gpu
            .text_encoder
            .debug_hidden_states(&gpu_token_ids, Some(&gpu_attention_mask));

        for (idx, (cpu_state, gpu_state)) in
            cpu_text_states.iter().zip(&gpu_text_states).enumerate()
        {
            let name = if idx + 1 == cpu_text_states.len() {
                "text_final_norm_output".to_string()
            } else if idx == 0 {
                "text_post_embedding_norm".to_string()
            } else {
                format!("text_post_layer_{}", idx - 1)
            };
            let _ = print_diff(&name, cpu_state, gpu_state).await.unwrap();
        }

        let text_config = load_modern_bert_config_from_gguf(&model_bytes);
        let cpu_layer2_input = cpu_text_states[2].clone();
        let gpu_layer2_input = gpu_text_states[2].clone();
        let cpu_layer2_attn_norm = load_layer_norm_from_gguf(
            &model_bytes,
            &["text", "blk.2", "attn_norm"],
            &cpu_device,
            text_config.norm_eps,
        );
        let gpu_layer2_attn_norm = load_layer_norm_from_gguf(
            &model_bytes,
            &["text", "blk.2", "attn_norm"],
            &gpu_device,
            text_config.norm_eps,
        );
        let cpu_layer2_attn_input = cpu_layer2_attn_norm.forward(&cpu_layer2_input);
        let gpu_layer2_attn_input = gpu_layer2_attn_norm.forward(&gpu_layer2_input);
        let _ = print_diff(
            "text_layer2_attn_norm_output",
            &cpu_layer2_attn_input,
            &gpu_layer2_attn_input,
        )
        .await
        .unwrap();

        let cpu_layer2_qkv = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.2"],
            "attn_qkv.weight",
            &cpu_device,
        );
        let gpu_layer2_qkv = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.2"],
            "attn_qkv.weight",
            &gpu_device,
        );
        let cpu_layer2_qkv_projection = cpu_layer2_attn_input.q_mat_mul(&cpu_layer2_qkv);
        let gpu_layer2_qkv_projection = gpu_layer2_attn_input.q_mat_mul(&gpu_layer2_qkv);
        let _ = print_diff(
            "text_layer2_qkv_projection",
            &cpu_layer2_qkv_projection,
            &gpu_layer2_qkv_projection,
        )
        .await
        .unwrap();
        let rope_cache_cpu = fusor::RopeCache::new(
            text_config.head_dimension,
            text_config.context_length,
            text_config.rope_theta,
            &cpu_device,
        )
        .unwrap();
        let rope_cache_gpu = fusor::RopeCache::new(
            text_config.head_dimension,
            text_config.context_length,
            text_config.rope_theta,
            &gpu_device,
        )
        .unwrap();

        let hidden_size = text_config.num_heads * text_config.head_dimension;
        let [b_sz, seq_len, _] = cpu_layer0_qkv_projection.shape();
        let cpu_query_states = cpu_layer0_qkv_projection
            .narrow(2, 0, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let cpu_key_states = cpu_layer0_qkv_projection
            .narrow(2, hidden_size, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_kv_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let cpu_value_states = cpu_layer0_qkv_projection
            .narrow(2, 2 * hidden_size, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_kv_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let gpu_query_states = gpu_layer0_qkv_projection
            .narrow(2, 0, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let gpu_key_states = gpu_layer0_qkv_projection
            .narrow(2, hidden_size, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_kv_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let gpu_value_states = gpu_layer0_qkv_projection
            .narrow(2, 2 * hidden_size, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_kv_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();

        let (cpu_query_after_rope, cpu_key_after_rope) =
            rope_cache_cpu.forward(&cpu_query_states, &cpu_key_states, 0);
        let (gpu_query_after_rope, gpu_key_after_rope) =
            rope_cache_gpu.forward(&gpu_query_states, &gpu_key_states, 0);
        let _ = print_diff(
            "text_layer0_query_after_rope",
            &cpu_query_after_rope,
            &gpu_query_after_rope,
        )
        .await
        .unwrap();
        let _ = print_diff(
            "text_layer0_key_after_rope",
            &cpu_key_after_rope,
            &gpu_key_after_rope,
        )
        .await
        .unwrap();

        let cpu_attention_scores =
            cpu_query_after_rope.mat_mul(&cpu_key_after_rope.transpose(2, 3));
        let gpu_attention_scores =
            gpu_query_after_rope.mat_mul(&gpu_key_after_rope.transpose(2, 3));
        let _ = print_diff(
            "text_layer0_attention_scores",
            &cpu_attention_scores,
            &gpu_attention_scores,
        )
        .await
        .unwrap();

        let scale = 1.0 / (text_config.head_dimension as f32).sqrt();
        let cpu_attention_probs = cpu_attention_scores
            .mul_scalar(scale)
            .softmax_last_dim::<3>();
        let gpu_attention_probs = gpu_attention_scores
            .mul_scalar(scale)
            .softmax_last_dim::<3>();
        let _ = print_diff(
            "text_layer0_attention_probs",
            &cpu_attention_probs,
            &gpu_attention_probs,
        )
        .await
        .unwrap();

        let cpu_attention_context = cpu_attention_probs.mat_mul(&cpu_value_states);
        let gpu_attention_context = gpu_attention_probs.mat_mul(&gpu_value_states);
        let _ = print_diff(
            "text_layer0_attention_context",
            &cpu_attention_context,
            &gpu_attention_context,
        )
        .await
        .unwrap();

        let cpu_attn_output_weight = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.0"],
            "attn_output.weight",
            &cpu_device,
        );
        let gpu_attn_output_weight = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.0"],
            "attn_output.weight",
            &gpu_device,
        );
        let cpu_attention_context_flat = cpu_attention_context
            .transpose(1, 2)
            .to_concrete()
            .reshape([b_sz, seq_len, hidden_size])
            .to_concrete();
        let gpu_attention_context_flat = gpu_attention_context
            .transpose(1, 2)
            .to_concrete()
            .reshape([b_sz, seq_len, hidden_size])
            .to_concrete();
        let cpu_attention_output = cpu_attention_context_flat.q_mat_mul(&cpu_attn_output_weight);
        let gpu_attention_output = gpu_attention_context_flat.q_mat_mul(&gpu_attn_output_weight);
        let _ = print_diff(
            "text_layer0_attention_output_projection",
            &cpu_attention_output,
            &gpu_attention_output,
        )
        .await
        .unwrap();

        let cpu_after_attention = cpu_text_after_embd_norm.add_(&cpu_attention_output);
        let gpu_after_attention = gpu_text_after_embd_norm.add_(&gpu_attention_output);
        let _ = print_diff(
            "text_layer0_after_attention_residual",
            &cpu_after_attention,
            &gpu_after_attention,
        )
        .await
        .unwrap();

        let cpu_ffn_norm = load_layer_norm_from_gguf(
            &model_bytes,
            &["text", "blk.0", "ffn_norm"],
            &cpu_device,
            text_config.norm_eps,
        );
        let gpu_ffn_norm = load_layer_norm_from_gguf(
            &model_bytes,
            &["text", "blk.0", "ffn_norm"],
            &gpu_device,
            text_config.norm_eps,
        );
        let cpu_ffn_input = cpu_ffn_norm.forward(&cpu_after_attention);
        let gpu_ffn_input = gpu_ffn_norm.forward(&gpu_after_attention);
        let _ = print_diff("text_layer0_ffn_input", &cpu_ffn_input, &gpu_ffn_input)
            .await
            .unwrap();

        let cpu_ffn_gate_up = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.0"],
            "ffn_gate_up.weight",
            &cpu_device,
        );
        let gpu_ffn_gate_up = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.0"],
            "ffn_gate_up.weight",
            &gpu_device,
        );
        let cpu_ffn_gate_up_proj = cpu_ffn_input.q_mat_mul(&cpu_ffn_gate_up).to_concrete();
        let gpu_ffn_gate_up_proj = gpu_ffn_input.q_mat_mul(&gpu_ffn_gate_up).to_concrete();
        let _ = print_diff(
            "text_layer0_ffn_gate_up_projection",
            &cpu_ffn_gate_up_proj,
            &gpu_ffn_gate_up_proj,
        )
        .await
        .unwrap();

        let intermediate_size = cpu_ffn_gate_up.shape()[0] / 2;
        let cpu_gate = cpu_ffn_gate_up_proj
            .narrow(2, 0, intermediate_size)
            .to_concrete();
        let cpu_up = cpu_ffn_gate_up_proj.narrow(2, intermediate_size, intermediate_size);
        let gpu_gate = gpu_ffn_gate_up_proj
            .narrow(2, 0, intermediate_size)
            .to_concrete();
        let gpu_up = gpu_ffn_gate_up_proj.narrow(2, intermediate_size, intermediate_size);
        let cpu_ffn_activated = cpu_gate.gelu().mul_(&cpu_up);
        let gpu_ffn_activated = gpu_gate.gelu().mul_(&gpu_up);
        let _ = print_diff(
            "text_layer0_ffn_activated",
            &cpu_ffn_activated,
            &gpu_ffn_activated,
        )
        .await
        .unwrap();

        let cpu_ffn_down = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.0"],
            "ffn_down.weight",
            &cpu_device,
        );
        let gpu_ffn_down = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.0"],
            "ffn_down.weight",
            &gpu_device,
        );
        let cpu_ffn_output = cpu_ffn_activated.q_mat_mul(&cpu_ffn_down);
        let gpu_ffn_output = gpu_ffn_activated.q_mat_mul(&gpu_ffn_down);
        let _ = print_diff("text_layer0_ffn_output", &cpu_ffn_output, &gpu_ffn_output)
            .await
            .unwrap();

        let cpu_layer0_output = cpu_after_attention.add_(&cpu_ffn_output);
        let gpu_layer0_output = gpu_after_attention.add_(&gpu_ffn_output);
        let _ = print_diff("text_layer0_output", &cpu_layer0_output, &gpu_layer0_output)
            .await
            .unwrap();

        let cpu_layer1_attn_norm = load_layer_norm_from_gguf(
            &model_bytes,
            &["text", "blk.1", "attn_norm"],
            &cpu_device,
            text_config.norm_eps,
        );
        let gpu_layer1_attn_norm = load_layer_norm_from_gguf(
            &model_bytes,
            &["text", "blk.1", "attn_norm"],
            &gpu_device,
            text_config.norm_eps,
        );
        let cpu_layer1_attn_input = cpu_layer1_attn_norm.forward(&cpu_layer0_output);
        let gpu_layer1_attn_input = gpu_layer1_attn_norm.forward(&gpu_layer0_output);
        let _ = print_diff(
            "text_layer1_attn_norm_output",
            &cpu_layer1_attn_input,
            &gpu_layer1_attn_input,
        )
        .await
        .unwrap();

        let cpu_layer2_query_states = cpu_layer2_qkv_projection
            .narrow(2, 0, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let cpu_layer2_key_states = cpu_layer2_qkv_projection
            .narrow(2, hidden_size, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_kv_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let cpu_layer2_value_states = cpu_layer2_qkv_projection
            .narrow(2, 2 * hidden_size, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_kv_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let gpu_layer2_query_states = gpu_layer2_qkv_projection
            .narrow(2, 0, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let gpu_layer2_key_states = gpu_layer2_qkv_projection
            .narrow(2, hidden_size, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_kv_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();
        let gpu_layer2_value_states = gpu_layer2_qkv_projection
            .narrow(2, 2 * hidden_size, hidden_size)
            .reshape([
                b_sz,
                seq_len,
                text_config.num_kv_heads,
                text_config.head_dimension,
            ])
            .transpose(1, 2)
            .to_concrete();

        let (cpu_layer2_query_after_rope, cpu_layer2_key_after_rope) =
            rope_cache_cpu.forward(&cpu_layer2_query_states, &cpu_layer2_key_states, 0);
        let (gpu_layer2_query_after_rope, gpu_layer2_key_after_rope) =
            rope_cache_gpu.forward(&gpu_layer2_query_states, &gpu_layer2_key_states, 0);
        let _ = print_diff(
            "text_layer2_query_after_rope",
            &cpu_layer2_query_after_rope,
            &gpu_layer2_query_after_rope,
        )
        .await
        .unwrap();
        let _ = print_diff(
            "text_layer2_key_after_rope",
            &cpu_layer2_key_after_rope,
            &gpu_layer2_key_after_rope,
        )
        .await
        .unwrap();

        let cpu_layer2_attention_scores =
            cpu_layer2_query_after_rope.mat_mul(&cpu_layer2_key_after_rope.transpose(2, 3));
        let gpu_layer2_attention_scores =
            gpu_layer2_query_after_rope.mat_mul(&gpu_layer2_key_after_rope.transpose(2, 3));
        let _ = print_diff(
            "text_layer2_attention_scores",
            &cpu_layer2_attention_scores,
            &gpu_layer2_attention_scores,
        )
        .await
        .unwrap();

        let cpu_layer2_attention_probs = cpu_layer2_attention_scores
            .mul_scalar(scale)
            .softmax_last_dim::<3>();
        let gpu_layer2_attention_probs = gpu_layer2_attention_scores
            .mul_scalar(scale)
            .softmax_last_dim::<3>();
        let _ = print_diff(
            "text_layer2_attention_probs",
            &cpu_layer2_attention_probs,
            &gpu_layer2_attention_probs,
        )
        .await
        .unwrap();

        let cpu_layer2_attention_context =
            cpu_layer2_attention_probs.mat_mul(&cpu_layer2_value_states);
        let gpu_layer2_attention_context =
            gpu_layer2_attention_probs.mat_mul(&gpu_layer2_value_states);
        let _ = print_diff(
            "text_layer2_attention_context",
            &cpu_layer2_attention_context,
            &gpu_layer2_attention_context,
        )
        .await
        .unwrap();

        let cpu_layer2_attn_output_weight = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.2"],
            "attn_output.weight",
            &cpu_device,
        );
        let gpu_layer2_attn_output_weight = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.2"],
            "attn_output.weight",
            &gpu_device,
        );
        let cpu_layer2_attention_context_flat = cpu_layer2_attention_context
            .transpose(1, 2)
            .to_concrete()
            .reshape([b_sz, seq_len, hidden_size])
            .to_concrete();
        let gpu_layer2_attention_context_flat = gpu_layer2_attention_context
            .transpose(1, 2)
            .to_concrete()
            .reshape([b_sz, seq_len, hidden_size])
            .to_concrete();
        let cpu_layer2_attention_output =
            cpu_layer2_attention_context_flat.q_mat_mul(&cpu_layer2_attn_output_weight);
        let gpu_layer2_attention_output =
            gpu_layer2_attention_context_flat.q_mat_mul(&gpu_layer2_attn_output_weight);
        let _ = print_diff(
            "text_layer2_attention_output_projection",
            &cpu_layer2_attention_output,
            &gpu_layer2_attention_output,
        )
        .await
        .unwrap();

        let cpu_layer2_after_attention = cpu_layer2_input.add_(&cpu_layer2_attention_output);
        let gpu_layer2_after_attention = gpu_layer2_input.add_(&gpu_layer2_attention_output);
        let _ = print_diff(
            "text_layer2_after_attention_residual",
            &cpu_layer2_after_attention,
            &gpu_layer2_after_attention,
        )
        .await
        .unwrap();

        let cpu_layer2_ffn_norm = load_layer_norm_from_gguf(
            &model_bytes,
            &["text", "blk.2", "ffn_norm"],
            &cpu_device,
            text_config.norm_eps,
        );
        let gpu_layer2_ffn_norm = load_layer_norm_from_gguf(
            &model_bytes,
            &["text", "blk.2", "ffn_norm"],
            &gpu_device,
            text_config.norm_eps,
        );
        let cpu_layer2_ffn_input = cpu_layer2_ffn_norm.forward(&cpu_layer2_after_attention);
        let gpu_layer2_ffn_input = gpu_layer2_ffn_norm.forward(&gpu_layer2_after_attention);
        let _ = print_diff(
            "text_layer2_ffn_input",
            &cpu_layer2_ffn_input,
            &gpu_layer2_ffn_input,
        )
        .await
        .unwrap();

        let cpu_layer2_ffn_gate_up = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.2"],
            "ffn_gate_up.weight",
            &cpu_device,
        );
        let gpu_layer2_ffn_gate_up = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.2"],
            "ffn_gate_up.weight",
            &gpu_device,
        );
        let cpu_layer2_ffn_gate_up_proj = cpu_layer2_ffn_input
            .q_mat_mul(&cpu_layer2_ffn_gate_up)
            .to_concrete();
        let gpu_layer2_ffn_gate_up_proj = gpu_layer2_ffn_input
            .q_mat_mul(&gpu_layer2_ffn_gate_up)
            .to_concrete();
        let _ = print_diff(
            "text_layer2_ffn_gate_up_projection",
            &cpu_layer2_ffn_gate_up_proj,
            &gpu_layer2_ffn_gate_up_proj,
        )
        .await
        .unwrap();

        let layer2_intermediate_size = cpu_layer2_ffn_gate_up.shape()[0] / 2;
        let cpu_layer2_gate = cpu_layer2_ffn_gate_up_proj
            .narrow(2, 0, layer2_intermediate_size)
            .to_concrete();
        let cpu_layer2_up = cpu_layer2_ffn_gate_up_proj.narrow(
            2,
            layer2_intermediate_size,
            layer2_intermediate_size,
        );
        let gpu_layer2_gate = gpu_layer2_ffn_gate_up_proj
            .narrow(2, 0, layer2_intermediate_size)
            .to_concrete();
        let gpu_layer2_up = gpu_layer2_ffn_gate_up_proj.narrow(
            2,
            layer2_intermediate_size,
            layer2_intermediate_size,
        );
        let cpu_layer2_ffn_activated = cpu_layer2_gate.gelu().mul_(&cpu_layer2_up);
        let gpu_layer2_ffn_activated = gpu_layer2_gate.gelu().mul_(&gpu_layer2_up);
        let _ = print_diff(
            "text_layer2_ffn_activated",
            &cpu_layer2_ffn_activated,
            &gpu_layer2_ffn_activated,
        )
        .await
        .unwrap();

        let cpu_layer2_ffn_down = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.2"],
            "ffn_down.weight",
            &cpu_device,
        );
        let gpu_layer2_ffn_down = load_qmatrix_tensor_from_gguf(
            &model_bytes,
            &["text", "blk.2"],
            "ffn_down.weight",
            &gpu_device,
        );
        let cpu_layer2_ffn_output = cpu_layer2_ffn_activated.q_mat_mul(&cpu_layer2_ffn_down);
        let gpu_layer2_ffn_output = gpu_layer2_ffn_activated.q_mat_mul(&gpu_layer2_ffn_down);
        let _ = print_diff(
            "text_layer2_ffn_output",
            &cpu_layer2_ffn_output,
            &gpu_layer2_ffn_output,
        )
        .await
        .unwrap();

        let cpu_layer2_output = cpu_layer2_after_attention.add_(&cpu_layer2_ffn_output);
        let gpu_layer2_output = gpu_layer2_after_attention.add_(&gpu_layer2_ffn_output);
        let _ = print_diff("text_layer2_output", &cpu_layer2_output, &gpu_layer2_output)
            .await
            .unwrap();

        let _ = print_diff(
            "token_embeddings",
            &cpu_token_embeddings,
            &gpu_token_embeddings,
        )
        .await
        .unwrap();

        let (cpu_word_embeddings, _) =
            first_subtoken_pooling(&cpu_token_embeddings, &[cpu_tokenized.clone()], &cpu_device);
        let (gpu_word_embeddings, _) =
            first_subtoken_pooling(&gpu_token_embeddings, &[gpu_tokenized.clone()], &gpu_device);
        let _ = print_diff(
            "word_embeddings",
            &cpu_word_embeddings,
            &gpu_word_embeddings,
        )
        .await
        .unwrap();

        let (cpu_span_embeddings, cpu_span_indices) =
            cpu.span_layer.forward(&cpu_word_embeddings, &cpu_device);
        let (gpu_span_embeddings, gpu_span_indices) =
            gpu.span_layer.forward(&gpu_word_embeddings, &gpu_device);
        assert_eq!(cpu_span_indices, gpu_span_indices);
        let _ = print_diff(
            "span_embeddings",
            &cpu_span_embeddings,
            &gpu_span_embeddings,
        )
        .await
        .unwrap();

        let cpu_scores = Scorer::forward(&cpu_span_embeddings, &cpu_label_embeddings);
        let gpu_scores = Scorer::forward(&gpu_span_embeddings, &gpu_label_embeddings);
        let max_score_diff = print_diff("span_scores", &cpu_scores, &gpu_scores)
            .await
            .unwrap();

        let cpu_entities = cpu.extract(text, &labels).await.unwrap();
        let gpu_entities = gpu.extract(text, &labels).await.unwrap();
        println!("cpu_entities={cpu_entities:?}");
        println!("gpu_entities={gpu_entities:?}");

        let cpu_entities: Vec<_> = cpu_entities
            .iter()
            .map(|entity| (entity.label.as_str(), entity.text.as_str()))
            .collect();
        let gpu_entities: Vec<_> = gpu_entities
            .iter()
            .map(|entity| (entity.label.as_str(), entity.text.as_str()))
            .collect();

        assert!(
            max_score_diff < 0.05,
            "CPU/GPU score drift is too large: max_abs_diff={max_score_diff:.6}"
        );
        assert_eq!(gpu_entities, cpu_entities);
    }
}
