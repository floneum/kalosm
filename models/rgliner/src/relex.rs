//! GLiNER-RelEx: Joint Named Entity Recognition and Relation Extraction.
//!
//! This module provides the `GlinerRelEx` struct for extracting entities and relations
//! from text using the GLiNER-RelEx model architecture.
//!
//! ## Architecture
//!
//! The model uses the following pipeline:
//! 1. mDeBERTa encoder for contextual embeddings
//! 2. BiLSTM for enhanced token representations
//! 3. Prompt representation layer for label embeddings
//! 4. Joint scorer for token-level BIO predictions
//! 5. Span layer for entity span representations
//! 6. Pair projector for relation classification
//!
//! ## Example
//!
//! ```rust,no_run
//! use rgliner::relex::*;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let mut relex = GlinerRelEx::builder()
//!     .with_source(GlinerRelExSource::relex_multi())
//!     .build()
//!     .await?;
//!
//! let (entities, relations) = relex.extract(
//!     "Apple was founded by Steve Jobs in California.",
//!     &["person", "organization", "location"],
//!     &["founded by", "located in"],
//! ).await?;
//!
//! for relation in relations {
//!     println!("{} --[{}]--> {}",
//!         relation.head.text,
//!         relation.relation,
//!         relation.tail.text
//!     );
//! }
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use fusor::{Device, Tensor, VarBuilder};
use kalosm_common::Cache;
use kalosm_model_types::{FileSource, ModelLoadingProgress};
use tokenizers::Tokenizer;

use crate::decoding::Entity;
use crate::error::{GlinerError, GlinerLoadingError};
use crate::raw::{BiLstm, JointScorer, PairProjector, PromptRepLayer, SpanLayer};
use crate::relation_decoding::Relation;
use crate::relex_tokenization::{RelExTokenizedInput, RelExTokenizer, SpecialTokenIds};
use rbert::raw::MDebertaModel;

/// Source configuration for GLiNER-RelEx models.
///
/// The GGUF file produced by `convert_relex_to_gguf.py` embeds the tokenizer
/// JSON and GLiNER config JSON as string metadata, so only the model file is
/// required. `tokenizer` and `config` can optionally override the embedded
/// copies (e.g., to swap in a custom tokenizer).
pub struct GlinerRelExSource {
    /// Main model GGUF file (encoder + all layers + embedded tokenizer/config)
    pub model: FileSource,
    /// Optional tokenizer JSON override. If `None`, the tokenizer is read from
    /// the `gliner.tokenizer_json` metadata embedded in the GGUF.
    pub tokenizer: Option<FileSource>,
    /// Optional GLiNER config JSON override. If `None`, the config is read from
    /// the `gliner.config_json` metadata embedded in the GGUF.
    pub config: Option<FileSource>,
}

impl GlinerRelExSource {
    /// Build a RelEx preset from its Demonthos GGUF filename. Tokenizer and
    /// config are read from the GGUF metadata, so both overrides start `None`.
    fn relex_variant(gguf_file: &str) -> Self {
        Self {
            model: FileSource::huggingface(
                "Demonthos/gliner-gguf".to_string(),
                "main".to_string(),
                gguf_file.to_string(),
            ),
            tokenizer: None,
            config: None,
        }
    }

    /// GLiNER-RelEx Multi v1.0 source.
    ///
    /// Multilingual variant built on `mdeberta-v3-base` with `span_mode = token_level`.
    /// Downloads the GGUF-converted weights from HuggingFace.
    ///
    /// Tokenizer and config are embedded in the GGUF file.
    pub fn relex_multi() -> Self {
        Self::relex_variant("gliner-relex-multi-v1.0-Q4_K.gguf")
    }

    /// GLiNER-RelEx Base v1.0 source.
    ///
    /// English-only variant built on `deberta-v3-base` with `span_mode = token_level`.
    /// Smaller than the multilingual variant but limited to English text.
    ///
    /// Tokenizer and config are embedded in the GGUF file.
    pub fn relex_base() -> Self {
        Self::relex_variant("gliner-relex-base-v1.0-Q4_K.gguf")
    }

    /// GLiNER-RelEx Large v1.0 source.
    ///
    /// English-only variant built on `deberta-v3-large` with `span_mode = markerV0`
    /// and a 1024→768 projection between the encoder and downstream heads.
    /// The most accurate variant but also the largest.
    ///
    /// Tokenizer and config are embedded in the GGUF file.
    pub fn relex_large() -> Self {
        Self::relex_variant("gliner-relex-large-v1.0-Q4_K.gguf")
    }

    /// Create a source from a local GGUF file.
    ///
    /// The tokenizer and config are expected to be embedded in the GGUF
    /// metadata (produced by `convert_relex_to_gguf.py`).
    pub fn local(model_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            model: FileSource::local(model_path.into()),
            tokenizer: None,
            config: None,
        }
    }

    /// Override the tokenizer source (otherwise read from GGUF metadata).
    pub fn with_tokenizer(mut self, tokenizer: FileSource) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Override the config source (otherwise read from GGUF metadata).
    pub fn with_config(mut self, config: FileSource) -> Self {
        self.config = Some(config);
        self
    }
}

impl Default for GlinerRelExSource {
    fn default() -> Self {
        Self::relex_multi()
    }
}

/// Configuration for GLiNER-RelEx model.
#[derive(Debug, Clone)]
pub struct GlinerRelExConfig {
    /// Maximum span width in words
    pub max_width: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Entity detection threshold
    pub entity_threshold: f32,
    /// Adjacency filtering threshold
    pub adjacency_threshold: f32,
    /// Relation classification threshold
    pub relation_threshold: f32,
    /// Special token IDs
    pub special_tokens: SpecialTokenIds,
}

impl Default for GlinerRelExConfig {
    fn default() -> Self {
        Self {
            max_width: 12,
            hidden_size: 768,
            entity_threshold: 0.4,
            adjacency_threshold: 0.55,
            relation_threshold: 0.8,
            special_tokens: SpecialTokenIds::default(),
        }
    }
}

/// Builder for constructing a [`GlinerRelEx`] model.
#[derive(Default)]
pub struct GlinerRelExBuilder {
    source: GlinerRelExSource,
    cache: Cache,
    device: Option<Device>,
    config: GlinerRelExConfig,
}

impl GlinerRelExBuilder {
    /// Set the model source.
    pub fn with_source(mut self, source: GlinerRelExSource) -> Self {
        self.source = source;
        self
    }

    /// Set the entity threshold.
    pub fn with_entity_threshold(mut self, threshold: f32) -> Self {
        self.config.entity_threshold = threshold;
        self
    }

    /// Set the adjacency threshold.
    pub fn with_adjacency_threshold(mut self, threshold: f32) -> Self {
        self.config.adjacency_threshold = threshold;
        self
    }

    /// Set the relation threshold.
    pub fn with_relation_threshold(mut self, threshold: f32) -> Self {
        self.config.relation_threshold = threshold;
        self
    }

    /// Set the maximum span width.
    pub fn with_max_width(mut self, max_width: usize) -> Self {
        self.config.max_width = max_width;
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
    pub async fn build(self) -> Result<GlinerRelEx, GlinerLoadingError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }

    /// Build the model with a loading handler.
    pub async fn build_with_loading_handler(
        self,
        loading_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> Result<GlinerRelEx, GlinerLoadingError> {
        GlinerRelEx::from_builder(self, loading_handler).await
    }
}

/// Span-scoring modes supported by the Rust inference path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanMode {
    /// Per-token BIO-style output (`[start, end, inside]` sigmoids per (token, label)).
    /// Used by the `multi` and `base` variants.
    TokenLevel,
    /// Per-span scoring: enumerate all spans up to `max_width`, score each against
    /// the projected entity prompts. Used by the `large` variants.
    MarkerV0,
}

/// GLiNER-RelEx model for joint NER and relation extraction.
pub struct GlinerRelEx {
    /// mDeBERTa encoder
    encoder: MDebertaModel,
    /// BiLSTM for enhanced token representations
    bilstm: BiLstm,
    /// Prompt representation layer for label projection
    prompt_rep_layer: PromptRepLayer,
    /// Joint scorer for token-level predictions (None for markerV0 variants).
    scorer: Option<JointScorer>,
    /// Span representation layer
    span_layer: SpanLayer,
    /// Entity pair projector
    pair_projector: PairProjector,
    /// Tokenizer with special token handling
    tokenizer: Arc<RelExTokenizer>,
    /// How entities are scored (derived from `gliner.span_mode` metadata).
    span_mode: SpanMode,
    /// Device
    device: Device,
    /// Configuration
    config: GlinerRelExConfig,
}

async fn default_device() -> Device {
    Device::gpu().await.unwrap_or_else(|_| Device::cpu())
}

impl GlinerRelEx {
    /// Create a new builder.
    pub fn builder() -> GlinerRelExBuilder {
        GlinerRelExBuilder::default()
    }

    /// Create with default settings.
    pub async fn new() -> Result<Self, GlinerLoadingError> {
        Self::builder().build().await
    }

    async fn from_builder(
        builder: GlinerRelExBuilder,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> Result<Self, GlinerLoadingError> {
        let GlinerRelExBuilder {
            source,
            cache,
            device,
            config,
        } = builder;

        // Download main model weights first - the GGUF may also contain the
        // tokenizer and config as embedded metadata.
        let model_bytes =
            crate::download_bytes(&cache, &source.model, "Model", &mut progress_handler).await?;

        // Initialize device
        let device = match device {
            Some(d) => d,
            None => default_device().await,
        };

        // Load model components from GGUF
        let mut model_cursor = std::io::Cursor::new(&model_bytes);
        let mut vb = VarBuilder::from_gguf(&mut model_cursor)
            .map_err(|err| GlinerLoadingError::LoadModel(fusor::Error::from(err)))?;

        // Determine span mode to pick the right decoder path. Supported modes
        // are `token_level` (base/multi) and `markerV0` (large).
        let span_mode_str = vb
            .get_metadata("gliner.span_mode")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "token_level".to_string());
        let span_mode = match span_mode_str.as_str() {
            "token_level" => SpanMode::TokenLevel,
            "markerV0" => SpanMode::MarkerV0,
            other => {
                return Err(GlinerLoadingError::LoadModel(fusor::Error::msg(format!(
                    "Unsupported gliner.span_mode '{other}'. \
                     Supported values: 'token_level', 'markerV0'."
                ))));
            }
        };

        // Resolve tokenizer: explicit override > embedded metadata.
        let tokenizer_bytes: Vec<u8> = if let Some(tokenizer_src) = source.tokenizer.as_ref() {
            crate::download_bytes(&cache, tokenizer_src, "Tokenizer", &mut progress_handler).await?
        } else {
            let meta = vb
                .get_metadata("gliner.tokenizer_json")
                .and_then(|v| v.to_string().ok())
                .ok_or_else(|| {
                    GlinerLoadingError::LoadModel(fusor::Error::msg(
                        "GGUF missing embedded tokenizer (metadata key `gliner.tokenizer_json`). \
                         Re-run convert_relex_to_gguf.py or set a tokenizer source via \
                         `GlinerRelExSource::with_tokenizer`.",
                    ))
                })?;
            meta.as_bytes().to_vec()
        };

        let tokenizer =
            Tokenizer::from_bytes(&tokenizer_bytes).map_err(GlinerLoadingError::LoadTokenizer)?;
        // Resolve special tokens from the tokenizer so we pick up the right IDs
        // regardless of variant (multi uses 250102/250103/250104, base/large
        // use 128001/128002/128003). Falls back to the user-supplied IDs.
        let mut effective_config = config;
        effective_config.special_tokens =
            SpecialTokenIds::from_tokenizer(&tokenizer, effective_config.special_tokens);
        let relex_tokenizer =
            RelExTokenizer::with_special_tokens(tokenizer, effective_config.special_tokens.clone());
        let config = effective_config;

        // Load encoder (mDeBERTa)
        let encoder = MDebertaModel::load(&device, &mut vb.pp("text"))?;

        // Load BiLSTM
        let bilstm = BiLstm::load(&device, &mut vb.pp("rnn"))?;

        // Load prompt representation layer
        let prompt_rep_layer = PromptRepLayer::load(&device, &mut vb.pp("prompt_rep_layer"))?;

        // Load joint scorer
        let scorer = match span_mode {
            SpanMode::TokenLevel => Some(JointScorer::load(&device, &mut vb.pp("scorer"))?),
            SpanMode::MarkerV0 => None,
        };

        // Load span layer
        let span_layer = SpanLayer::load(&device, &mut vb)?;

        // Load pair projector
        let pair_projector = PairProjector::load(&device, &mut vb.pp("pair_proj"))?;

        Ok(Self {
            encoder,
            bilstm,
            prompt_rep_layer,
            scorer,
            span_layer,
            pair_projector,
            tokenizer: Arc::new(relex_tokenizer),
            span_mode,
            device,
            config,
        })
    }

    /// Extract entities and relations from text.
    ///
    /// # Arguments
    /// * `text` - Input text
    /// * `entity_labels` - Entity type labels (e.g., ["person", "organization"])
    /// * `relation_labels` - Relation type labels (e.g., ["founded by", "works at"])
    ///
    /// # Returns
    /// Tuple of (entities, relations)
    pub async fn extract(
        &self,
        text: &str,
        entity_labels: &[&str],
        relation_labels: &[&str],
    ) -> Result<(Vec<Entity>, Vec<Relation>), GlinerError> {
        let mut results = self.extract_batch(&[text], entity_labels, relation_labels).await?;
        Ok(results.pop().unwrap_or_default())
    }

    /// Extract entities and relations from a batch of texts.
    pub async fn extract_batch(
        &self,
        texts: &[&str],
        entity_labels: &[&str],
        relation_labels: &[&str],
    ) -> Result<Vec<(Vec<Entity>, Vec<Relation>)>, GlinerError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let tokenized = self
            .tokenizer
            .tokenize_batch(texts, entity_labels, relation_labels)?;
        let (token_ids, attention_mask) = self.build_batched_inputs(&tokenized);
        let encoder_output = self.encoder.forward(&token_ids, Some(&attention_mask));

        let text_positions: Vec<Vec<usize>> = tokenized
            .iter()
            .map(|item| item.text_positions.clone())
            .collect();
        let word_encoder_embs =
            self.gather_at_positions_batched(&encoder_output, &text_positions);
        let word_lengths: Vec<usize> = tokenized.iter().map(|item| item.num_words).collect();
        let text_embs = self
            .bilstm
            .forward_with_lengths(&word_encoder_embs, &word_lengths)
            .await;

        let ent_positions: Vec<Vec<usize>> = tokenized
            .iter()
            .map(|item| item.ent_positions.clone())
            .collect();
        let ent_embs_raw = self.gather_at_positions_batched(&encoder_output, &ent_positions);
        let ent_embs = self.prompt_rep_layer.forward_3d(&ent_embs_raw);

        let rel_positions: Vec<Vec<usize>> = tokenized
            .iter()
            .map(|item| item.rel_positions.clone())
            .collect();
        let rel_embs = self.gather_at_positions_batched(&encoder_output, &rel_positions);

        let entities_per_item = match self.span_mode {
            SpanMode::TokenLevel => {
                let scorer = self.scorer.as_ref().expect("token_level requires scorer");
                let token_scores = scorer.forward_entity_scores(&text_embs, &ent_embs);
                self.decode_entities_from_tokens_batch(
                    &token_scores,
                    entity_labels,
                    &tokenized,
                    texts,
                )
                .await?
            }
            SpanMode::MarkerV0 => {
                self.decode_entities_marker_v0_batch(
                    &text_embs,
                    &ent_embs,
                    entity_labels,
                    &tokenized,
                    texts,
                )
                .await?
            }
        };

        let mut results = Vec::with_capacity(texts.len());
        for (batch_idx, entities) in entities_per_item.into_iter().enumerate() {
            let relations = if entities.len() < 2 || relation_labels.is_empty() {
                Vec::new()
            } else {
                let text_embs_item: Tensor<3, f32> =
                    text_embs.narrow(0, batch_idx, 1).to_concrete();
                let rel_embs_item: Tensor<3, f32> =
                    rel_embs.narrow(0, batch_idx, 1).to_concrete();
                self.decode_relations(
                    &text_embs_item,
                    &rel_embs_item,
                    &entities,
                    relation_labels,
                )
                .await?
            };
            results.push((entities, relations));
        }

        Ok(results)
    }

    /// Extract entities and relations from text, chunking the input first so long
    /// documents that would otherwise be truncated by the encoder's context window
    /// still get full coverage.
    ///
    /// Uses the model's own tokenizer to pack whole words into chunks of at most
    /// `token_budget` subtokens with ~15% overlap between adjacent chunks. Each
    /// chunk is scored independently; entity and relation byte offsets are remapped
    /// back into the original text and deduped across overlapping windows (keeping
    /// the highest score per span+label / head+tail+label).
    ///
    /// `token_budget` defaults to 128.
    pub async fn extract_auto(
        &self,
        text: &str,
        entity_labels: &[&str],
        relation_labels: &[&str],
        token_budget: Option<usize>,
    ) -> Result<(Vec<Entity>, Vec<Relation>), GlinerError> {
        let budget = token_budget.unwrap_or(128);
        let ranges = crate::tokenization::token_packed_ranges(
            self.tokenizer.tokenizer(),
            text,
            budget,
            budget / 7,
        )?;
        if ranges.len() <= 1 {
            return self.extract(text, entity_labels, relation_labels).await;
        }

        let shift = |ent: &mut Entity, offset: usize| {
            ent.start_char += offset;
            ent.end_char += offset;
        };

        let chunk_texts: Vec<&str> = ranges.iter().map(|range| &text[range.clone()]).collect();
        let per_chunk = self
            .extract_batch(&chunk_texts, entity_labels, relation_labels)
            .await?;

        let mut all_entities: Vec<Entity> = Vec::new();
        let mut all_relations: Vec<Relation> = Vec::new();
        for (range, (entities, relations)) in ranges.iter().zip(per_chunk) {
            let offset = range.start;
            for mut ent in entities {
                shift(&mut ent, offset);
                all_entities.push(ent);
            }
            for mut rel in relations {
                shift(&mut rel.head, offset);
                shift(&mut rel.tail, offset);
                all_relations.push(rel);
            }
        }

        all_entities.sort_by(|a, b| {
            a.start_char
                .cmp(&b.start_char)
                .then_with(|| a.end_char.cmp(&b.end_char))
                .then_with(|| a.label.cmp(&b.label))
        });
        all_entities.dedup_by(|b, a| {
            if a.start_char == b.start_char && a.end_char == b.end_char && a.label == b.label {
                if b.score > a.score {
                    a.score = b.score;
                }
                true
            } else {
                false
            }
        });

        all_relations.sort_by(|a, b| {
            a.head
                .start_char
                .cmp(&b.head.start_char)
                .then_with(|| a.tail.start_char.cmp(&b.tail.start_char))
                .then_with(|| a.relation.cmp(&b.relation))
        });
        all_relations.dedup_by(|b, a| {
            if a.head.start_char == b.head.start_char
                && a.head.end_char == b.head.end_char
                && a.tail.start_char == b.tail.start_char
                && a.tail.end_char == b.tail.end_char
                && a.relation == b.relation
            {
                if b.score > a.score {
                    a.score = b.score;
                }
                true
            } else {
                false
            }
        });

        Ok((all_entities, all_relations))
    }

    fn build_batched_inputs(
        &self,
        tokenized: &[RelExTokenizedInput],
    ) -> (Tensor<2, u32>, Tensor<2, u32>) {
        let batch_size = tokenized.len();
        let max_seq_len = tokenized
            .iter()
            .map(|item| item.token_ids.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let pad_id = self.tokenizer.special_tokens().pad_id;

        let mut token_ids = vec![pad_id; batch_size * max_seq_len];
        let mut attention_mask = vec![0u32; batch_size * max_seq_len];
        for (batch_idx, item) in tokenized.iter().enumerate() {
            let offset = batch_idx * max_seq_len;
            let len = item.token_ids.len();
            token_ids[offset..offset + len].copy_from_slice(&item.token_ids);
            attention_mask[offset..offset + len].copy_from_slice(&item.attention_mask);
        }

        let token_ids = Tensor::new(&self.device, &token_ids)
            .reshape([batch_size, max_seq_len])
            .to_concrete();
        let attention_mask = Tensor::new(&self.device, &attention_mask)
            .reshape([batch_size, max_seq_len])
            .to_concrete();
        (token_ids, attention_mask)
    }

    async fn decode_relations(
        &self,
        text_embs: &Tensor<3, f32>,
        rel_embs: &Tensor<3, f32>,
        entities: &[Entity],
        relation_labels: &[&str],
    ) -> Result<Vec<Relation>, GlinerError> {
        if entities.len() < 2 || relation_labels.is_empty() {
            return Ok(Vec::new());
        }

        let entity_spans: Vec<(usize, usize)> = entities
            .iter()
            .map(|e| (e.start_word, e.end_word))
            .collect();
        let span_reps = self
            .span_layer
            .forward_for_spans(text_embs, &entity_spans, &self.device);

        let num_entities = entities.len();
        let hidden_size = self.config.hidden_size;
        let mut candidate_pairs: Vec<(usize, usize)> = Vec::new();
        for head in 0..num_entities {
            for tail in 0..num_entities {
                if head != tail {
                    candidate_pairs.push((head, tail));
                }
            }
        }

        let span_reps_data = span_reps.clone().as_slice().await?;
        let span_reps_slice = span_reps_data.as_slice();
        let mut head_embs = Vec::with_capacity(candidate_pairs.len() * hidden_size);
        let mut tail_embs = Vec::with_capacity(candidate_pairs.len() * hidden_size);
        for &(head_idx, tail_idx) in &candidate_pairs {
            let h_start = head_idx * hidden_size;
            let t_start = tail_idx * hidden_size;
            head_embs.extend_from_slice(&span_reps_slice[h_start..h_start + hidden_size]);
            tail_embs.extend_from_slice(&span_reps_slice[t_start..t_start + hidden_size]);
        }

        let head_tensor = Tensor::new(&self.device, &head_embs)
            .reshape([candidate_pairs.len(), hidden_size])
            .to_concrete();
        let tail_tensor = Tensor::new(&self.device, &tail_embs)
            .reshape([candidate_pairs.len(), hidden_size])
            .to_concrete();
        let pair_embs = self.pair_projector.forward(&head_tensor, &tail_tensor);

        let rel_embs_squeezed: Tensor<2, f32> = rel_embs.squeeze(0).to_concrete();
        let rel_scores = pair_embs.mat_mul(&rel_embs_squeezed.transpose(0, 1));
        let rel_scores_slice = rel_scores.clone().as_slice().await?;
        let n_rels = relation_labels.len();
        let threshold = self.config.relation_threshold;

        let mut relations = Vec::new();
        for (pair_idx, &(head_idx, tail_idx)) in candidate_pairs.iter().enumerate() {
            let base = pair_idx * n_rels;
            for rel_idx in 0..n_rels {
                let raw = rel_scores_slice.as_slice()[base + rel_idx];
                let prob = 1.0 / (1.0 + (-raw).exp());
                if prob > threshold {
                    relations.push(Relation {
                        head: entities[head_idx].clone(),
                        tail: entities[tail_idx].clone(),
                        relation: relation_labels[rel_idx].to_string(),
                        score: prob,
                    });
                }
            }
        }

        relations.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(relations)
    }

    async fn decode_entities_marker_v0_batch(
        &self,
        text_embs: &Tensor<3, f32>,
        ent_embs: &Tensor<3, f32>,
        entity_labels: &[&str],
        tokenized: &[RelExTokenizedInput],
        texts: &[&str],
    ) -> Result<Vec<Vec<Entity>>, GlinerError> {
        let spans_per_batch: Vec<Vec<(usize, usize)>> = tokenized
            .iter()
            .map(|item| {
                let mut spans = Vec::new();
                for start in 0..item.num_words {
                    for width in 1..=self.config.max_width.min(item.num_words - start) {
                        spans.push((start, start + width - 1));
                    }
                }
                spans
            })
            .collect();

        let (flat_span_reps, span_counts) =
            self.span_layer
                .forward_for_spans_batched(text_embs, &spans_per_batch, &self.device);

        let mut offset = 0usize;
        let mut results = Vec::with_capacity(tokenized.len());
        for batch_idx in 0..tokenized.len() {
            let span_count = span_counts[batch_idx];
            let entities = if span_count == 0 || entity_labels.is_empty() {
                Vec::new()
            } else {
                let span_reps: Tensor<2, f32> =
                    flat_span_reps.narrow(0, offset, span_count).to_concrete();
                let ent_embs_2d: Tensor<2, f32> =
                    ent_embs.narrow(0, batch_idx, 1).squeeze(0).to_concrete();
                self.decode_entities_marker_v0_from_span_reps(
                    &span_reps,
                    &spans_per_batch[batch_idx],
                    &ent_embs_2d,
                    entity_labels,
                    &tokenized[batch_idx].word_offsets,
                    texts[batch_idx],
                )
                .await?
            };
            results.push(entities);
            offset += span_count;
        }

        Ok(results)
    }

    /// Resolve scored span candidates `(start_word, end_word, label_idx, score)`
    /// into entities: greedy highest-score-first non-overlapping selection, then
    /// sort the survivors by score descending. Shared by the marker-v0 and
    /// token-level decoders, which differ only in how they produce candidates.
    fn resolve_entity_candidates(
        &self,
        mut candidates: Vec<(usize, usize, usize, f32)>,
        entity_labels: &[&str],
        word_offsets: &[(usize, usize)],
        text: &str,
    ) -> Vec<Entity> {
        candidates.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        let mut taken: Vec<(usize, usize)> = Vec::new();
        let mut entities = Vec::new();
        for (s, e, l, score) in candidates {
            let overlap = taken.iter().any(|&(a, b)| !(e < a || s > b));
            if overlap {
                continue;
            }
            taken.push((s, e));
            if s < word_offsets.len() && e < word_offsets.len() {
                let (start_char, _) = word_offsets[s];
                let (_, end_char) = word_offsets[e];
                entities.push(Entity {
                    text: text[start_char..end_char].to_string(),
                    label: entity_labels[l].to_string(),
                    start_char,
                    end_char,
                    start_word: s,
                    end_word: e,
                    score,
                });
            }
        }

        entities.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entities
    }

    async fn decode_entities_marker_v0_from_span_reps(
        &self,
        span_reps: &Tensor<2, f32>,
        spans: &[(usize, usize)],
        ent_embs_2d: &Tensor<2, f32>,
        entity_labels: &[&str],
        word_offsets: &[(usize, usize)],
        text: &str,
    ) -> Result<Vec<Entity>, GlinerError> {
        let threshold = self.config.entity_threshold;
        let n_labels = entity_labels.len();
        if spans.is_empty() || n_labels == 0 {
            return Ok(Vec::new());
        }

        let label_rep_t: Tensor<2, f32> = ent_embs_2d.transpose(0, 1).to_concrete();
        let logits = span_reps.mat_mul(&label_rep_t);
        let logits_data = logits.clone().as_slice().await?;
        let logits_slice = logits_data.as_slice();

        let mut candidates: Vec<(usize, usize, usize, f32)> = Vec::new();
        for (span_idx, &(s, e)) in spans.iter().enumerate() {
            for l in 0..n_labels {
                let raw = logits_slice[span_idx * n_labels + l];
                let prob = 1.0 / (1.0 + (-raw).exp());
                if prob >= threshold {
                    candidates.push((s, e, l, prob));
                }
            }
        }

        Ok(self.resolve_entity_candidates(candidates, entity_labels, word_offsets, text))
    }

    async fn decode_entities_from_tokens_batch(
        &self,
        token_scores: &Tensor<4, f32>,
        entity_labels: &[&str],
        tokenized: &[RelExTokenizedInput],
        texts: &[&str],
    ) -> Result<Vec<Vec<Entity>>, GlinerError> {
        let [batch_size, padded_tokens, num_labels, num_channels] = token_scores.shape();
        assert_eq!(num_channels, 3, "expected [start, end, inside]");
        assert_eq!(batch_size, tokenized.len(), "tokenized batch size mismatch");
        let scores_data = token_scores.clone().as_slice().await?;
        let scores = scores_data.as_slice();
        let batch_stride = padded_tokens * num_labels * 3;

        let mut results = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            let start = batch_idx * batch_stride;
            let end = start + batch_stride;
            results.push(self.decode_entities_from_tokens_slice(
                &scores[start..end],
                tokenized[batch_idx].num_words,
                num_labels,
                entity_labels,
                &tokenized[batch_idx].word_offsets,
                texts[batch_idx],
            ));
        }
        Ok(results)
    }

    fn decode_entities_from_tokens_slice(
        &self,
        scores: &[f32],
        num_tokens: usize,
        num_labels: usize,
        entity_labels: &[&str],
        word_offsets: &[(usize, usize)],
        text: &str,
    ) -> Vec<Entity> {
        let threshold = self.config.entity_threshold;
        let mut candidates: Vec<(usize, usize, usize, f32)> = Vec::new();

        let score_at = |tok: usize, lab: usize, ch: usize| -> f32 {
            scores[tok * num_labels * 3 + lab * 3 + ch]
        };

        for label_idx in 0..num_labels {
            for start_tok in 0..num_tokens {
                let start_score = score_at(start_tok, label_idx, 0);
                if start_score < threshold {
                    continue;
                }

                for end_tok in start_tok..num_tokens {
                    let end_score = score_at(end_tok, label_idx, 1);
                    if end_score < threshold {
                        continue;
                    }

                    let mut min_score = start_score.min(end_score);
                    let mut valid = true;
                    for t in start_tok..=end_tok {
                        let inside = score_at(t, label_idx, 2);
                        if inside < threshold {
                            valid = false;
                            break;
                        }
                        if inside < min_score {
                            min_score = inside;
                        }
                    }
                    if valid {
                        candidates.push((start_tok, end_tok, label_idx, min_score));
                    }
                }
            }
        }

        self.resolve_entity_candidates(candidates, entity_labels, word_offsets, text)
    }

    /// Gather hidden states at specific positions for a whole batch.
    fn gather_at_positions_batched(
        &self,
        hidden_states: &Tensor<3, f32>,
        positions_per_batch: &[Vec<usize>],
    ) -> Tensor<3, f32> {
        let [batch_size, seq_len, hidden_size] = hidden_states.shape();
        assert_eq!(
            batch_size,
            positions_per_batch.len(),
            "positions_per_batch must match batch size"
        );

        let max_positions = positions_per_batch.iter().map(Vec::len).max().unwrap_or(0);
        if max_positions == 0 {
            return Tensor::zeros(&self.device, [batch_size, 1, hidden_size]);
        }

        let hidden_flat = hidden_states
            .to_concrete()
            .reshape([batch_size * seq_len, hidden_size])
            .to_concrete();

        let mut offset_indices = Vec::with_capacity(batch_size * max_positions);
        for (batch_idx, positions) in positions_per_batch.iter().enumerate() {
            let offset = (batch_idx * seq_len) as u32;
            for pos_idx in 0..max_positions {
                let pos = positions.get(pos_idx).copied().unwrap_or(0) as u32;
                offset_indices.push(pos + offset);
            }
        }

        let offset_indices = Tensor::new(&self.device, &offset_indices);
        hidden_flat
            .index_select(0, &offset_indices)
            .reshape([batch_size, max_positions, hidden_size])
            .to_concrete()
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the configuration.
    pub fn config(&self) -> &GlinerRelExConfig {
        &self.config
    }
}
