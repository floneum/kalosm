//! RelEx tokenization with special token handling.
//!
//! Builds joint input sequences in the format:
//! `[CLS] <<ENT>> label1 <<ENT>> label2 <<SEP>> <<REL>> rel1 <<REL>> rel2 <<SEP>> text... [SEP]`

use crate::error::GlinerError;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Special token IDs for GLiNER-RelEx.
#[derive(Debug, Clone)]
pub struct SpecialTokenIds {
    /// [CLS] token ID.
    pub cls_id: u32,
    /// [SEP] token ID (end-of-sequence separator).
    pub sep_id: u32,
    /// [PAD] token ID.
    pub pad_id: u32,
    /// <<ENT>> token ID (entity marker).
    pub ent_id: u32,
    /// <<REL>> token ID (relation marker).
    pub rel_id: u32,
    /// <<SEP>> token ID (internal separator between entity/relation/text blocks).
    pub inner_sep_id: u32,
}

impl Default for SpecialTokenIds {
    fn default() -> Self {
        // Defaults match mdeberta-v3-base (used by gliner-relex-multi-v1.0).
        // For other variants (deberta-v3-base/large), use
        // `SpecialTokenIds::from_tokenizer` to resolve the IDs dynamically.
        Self {
            cls_id: 1,
            sep_id: 2,
            pad_id: 0,
            ent_id: 250102,
            rel_id: 250104,
            inner_sep_id: 250103,
        }
    }
}

impl SpecialTokenIds {
    /// Resolve IDs by querying the tokenizer for each special token.
    ///
    /// This handles vocab differences between variants (e.g. multi vs base/large
    /// where `<<ENT>>` is id 250102 vs 128001). Falls back to the corresponding
    /// field in `fallback` if the tokenizer doesn't contain a particular token.
    pub fn from_tokenizer(tokenizer: &tokenizers::Tokenizer, fallback: Self) -> Self {
        let lookup =
            |tok: &str, default: u32| -> u32 { tokenizer.token_to_id(tok).unwrap_or(default) };
        Self {
            cls_id: lookup("[CLS]", fallback.cls_id),
            sep_id: lookup("[SEP]", fallback.sep_id),
            pad_id: lookup("[PAD]", fallback.pad_id),
            ent_id: lookup("<<ENT>>", fallback.ent_id),
            rel_id: lookup("<<REL>>", fallback.rel_id),
            inner_sep_id: lookup("<<SEP>>", fallback.inner_sep_id),
        }
    }
}

/// Tokenized RelEx input with position tracking.
#[derive(Debug, Clone)]
pub struct RelExTokenizedInput {
    /// Token IDs for the full sequence
    pub token_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<u32>,
    /// Positions of <<ENT>> tokens (indices into token_ids)
    pub ent_positions: Vec<usize>,
    /// Positions of <<REL>> tokens (indices into token_ids)
    pub rel_positions: Vec<usize>,
    /// Positions of first subtoken for each text word (indices into token_ids)
    pub text_positions: Vec<usize>,
    /// Word offsets in the original text (start_char, end_char)
    pub word_offsets: Vec<(usize, usize)>,
    /// Number of text words
    pub num_words: usize,
    /// Number of entity labels
    pub num_entity_labels: usize,
    /// Number of relation labels
    pub num_relation_labels: usize,
}

/// RelEx tokenizer for building joint input sequences.
pub struct RelExTokenizer {
    tokenizer: Arc<Tokenizer>,
    special_tokens: SpecialTokenIds,
}

impl RelExTokenizer {
    /// Create a new RelEx tokenizer.
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            special_tokens: SpecialTokenIds::default(),
        }
    }

    /// Create with custom special token IDs.
    pub fn with_special_tokens(tokenizer: Tokenizer, special_tokens: SpecialTokenIds) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            special_tokens,
        }
    }

    /// Tokenize text, entity labels, and relation labels into a joint sequence.
    ///
    /// Output format (matches Python GLiNER):
    /// `[CLS] <<ENT>> label1 <<ENT>> label2 <<SEP>> <<REL>> rel1 <<REL>> rel2 <<SEP>> word1 word2 ... [SEP]`
    ///
    /// Key details:
    /// - `<<SEP>>` (inner_sep_id) is used between entity/relation/text blocks (not `[SEP]`)
    /// - `[SEP]` is used only at the end of the sequence
    /// - Each word in the text is tokenized independently so that SentencePiece adds the
    ///   leading `▁` marker for each word.
    pub fn tokenize(
        &self,
        text: &str,
        entity_labels: &[&str],
        relation_labels: &[&str],
    ) -> Result<RelExTokenizedInput, GlinerError> {
        let mut token_ids = Vec::new();
        let mut ent_positions = Vec::new();
        let mut rel_positions = Vec::new();
        let mut text_positions = Vec::new();
        let mut word_offsets = Vec::new();

        // Start with [CLS]
        token_ids.push(self.special_tokens.cls_id);

        // Encode entity labels block: <<ENT>> label1 <<ENT>> label2 ...
        for label in entity_labels {
            ent_positions.push(token_ids.len());
            token_ids.push(self.special_tokens.ent_id);

            let label_encoding = self
                .tokenizer
                .encode(label.to_string(), false)
                .map_err(|e| GlinerError::TokenizationError(e.to_string()))?;
            token_ids.extend(label_encoding.get_ids().iter().copied());
        }
        // Internal separator
        token_ids.push(self.special_tokens.inner_sep_id);

        // Encode relation labels block: <<REL>> rel1 <<REL>> rel2 ...
        for label in relation_labels {
            rel_positions.push(token_ids.len());
            token_ids.push(self.special_tokens.rel_id);

            let label_encoding = self
                .tokenizer
                .encode(label.to_string(), false)
                .map_err(|e| GlinerError::TokenizationError(e.to_string()))?;
            token_ids.extend(label_encoding.get_ids().iter().copied());
        }
        // Internal separator between relations and text
        token_ids.push(self.special_tokens.inner_sep_id);

        // Encode text with word-level tracking: each word is tokenized separately
        let words = crate::tokenization::split_words(text);
        for (word, (start_char, end_char)) in words {
            text_positions.push(token_ids.len());
            word_offsets.push((start_char, end_char));

            let word_encoding = self
                .tokenizer
                .encode(word.to_string(), false)
                .map_err(|e| GlinerError::TokenizationError(e.to_string()))?;
            token_ids.extend(word_encoding.get_ids().iter().copied());
        }

        // Final [SEP]
        token_ids.push(self.special_tokens.sep_id);

        let num_words = text_positions.len();
        let attention_mask = vec![1u32; token_ids.len()];

        Ok(RelExTokenizedInput {
            token_ids,
            attention_mask,
            ent_positions,
            rel_positions,
            text_positions,
            word_offsets,
            num_words,
            num_entity_labels: entity_labels.len(),
            num_relation_labels: relation_labels.len(),
        })
    }

    /// Tokenize a batch of texts with a shared label prompt.
    pub fn tokenize_batch(
        &self,
        texts: &[&str],
        entity_labels: &[&str],
        relation_labels: &[&str],
    ) -> Result<Vec<RelExTokenizedInput>, GlinerError> {
        texts.iter()
            .map(|text| self.tokenize(text, entity_labels, relation_labels))
            .collect()
    }

    /// Get the underlying tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get special token IDs.
    pub fn special_tokens(&self) -> &SpecialTokenIds {
        &self.special_tokens
    }
}
