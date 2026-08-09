//! RelEx tokenization with special token handling.
//!
//! Builds joint input sequences in the format:
//! `[CLS] <<ENT>> label1 <<ENT>> label2 <<SEP>> <<REL>> rel1 <<REL>> rel2 <<SEP>> text... [SEP]`

use crate::error::{GlinerError, GlinerLoadingError};
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
    /// where `<<ENT>>` is id 250102 vs 128001).
    ///
    /// The standard tokens (`[CLS]`, `[SEP]`, `[PAD]`) fall back to the
    /// corresponding field in `fallback` when the tokenizer doesn't name them —
    /// their conventional IDs are stable across DeBERTa variants. The
    /// GLiNER-RelEx markers (`<<ENT>>`, `<<REL>>`, `<<SEP>>`), however, are
    /// **required**: silently falling back here would emit IDs from the wrong
    /// vocabulary (e.g. the ~250k mDeBERTa IDs against a 128k DeBERTa model),
    /// producing out-of-range embedding lookups and garbage scores. If any
    /// marker is missing we return [`GlinerLoadingError::MissingSpecialToken`]
    /// so loading fails loudly instead.
    pub fn from_tokenizer(
        tokenizer: &tokenizers::Tokenizer,
        fallback: Self,
    ) -> Result<Self, GlinerLoadingError> {
        let optional =
            |tok: &str, default: u32| -> u32 { tokenizer.token_to_id(tok).unwrap_or(default) };

        let (Some(ent_id), Some(rel_id), Some(inner_sep_id)) = (
            tokenizer.token_to_id("<<ENT>>"),
            tokenizer.token_to_id("<<REL>>"),
            tokenizer.token_to_id("<<SEP>>"),
        ) else {
            let missing: Vec<&str> = ["<<ENT>>", "<<REL>>", "<<SEP>>"]
                .into_iter()
                .filter(|tok| tokenizer.token_to_id(tok).is_none())
                .collect();
            return Err(GlinerLoadingError::MissingSpecialToken(missing.join(", ")));
        };

        Ok(Self {
            cls_id: optional("[CLS]", fallback.cls_id),
            sep_id: optional("[SEP]", fallback.sep_id),
            pad_id: optional("[PAD]", fallback.pad_id),
            ent_id,
            rel_id,
            inner_sep_id,
        })
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
                .map_err(GlinerError::Tokenizer)?;
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
                .map_err(GlinerError::Tokenizer)?;
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
                .map_err(GlinerError::Tokenizer)?;
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
        texts
            .iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an in-memory WordLevel tokenizer whose vocab is exactly `pairs`.
    /// Only `token_to_id` is exercised, so no normalizer/pre-tokenizer is needed.
    fn tokenizer_with(pairs: &[(&str, u32)]) -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (tok, id) in pairs {
            vocab.insert((*tok).to_string(), serde_json::json!(*id));
        }
        let tok_json = serde_json::json!({
            "version": "1.0",
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]" }
        });
        Tokenizer::from_bytes(tok_json.to_string().as_bytes()).expect("valid tokenizer json")
    }

    #[test]
    fn from_tokenizer_resolves_marker_ids() {
        // Multi (mdeberta) vocab layout.
        let tok = tokenizer_with(&[
            ("[CLS]", 1),
            ("[SEP]", 2),
            ("[PAD]", 0),
            ("<<ENT>>", 250102),
            ("<<REL>>", 250104),
            ("<<SEP>>", 250103),
        ]);
        let ids = SpecialTokenIds::from_tokenizer(&tok, SpecialTokenIds::default()).unwrap();
        assert_eq!(ids.cls_id, 1);
        assert_eq!(ids.sep_id, 2);
        assert_eq!(ids.pad_id, 0);
        assert_eq!(ids.ent_id, 250102);
        assert_eq!(ids.rel_id, 250104);
        assert_eq!(ids.inner_sep_id, 250103);
    }

    #[test]
    fn from_tokenizer_resolves_base_variant_without_leaking_multi_defaults() {
        // base/large (deberta) vocab: markers at 128001-128003. The fallback is
        // the multi default (250102/...), which must NOT leak through.
        let tok = tokenizer_with(&[
            ("[CLS]", 1),
            ("[SEP]", 2),
            ("[PAD]", 0),
            ("<<ENT>>", 128001),
            ("<<REL>>", 128003),
            ("<<SEP>>", 128002),
        ]);
        let ids = SpecialTokenIds::from_tokenizer(&tok, SpecialTokenIds::default()).unwrap();
        assert_eq!(ids.ent_id, 128001);
        assert_eq!(ids.rel_id, 128003);
        assert_eq!(ids.inner_sep_id, 128002);
    }

    #[test]
    fn from_tokenizer_falls_back_for_standard_tokens_only() {
        // Markers present, but [CLS]/[SEP]/[PAD] absent -> use fallback for the
        // standard tokens (no error), markers still come from the tokenizer.
        let tok = tokenizer_with(&[("<<ENT>>", 11), ("<<REL>>", 13), ("<<SEP>>", 12)]);
        let fallback = SpecialTokenIds {
            cls_id: 7,
            sep_id: 8,
            pad_id: 9,
            ..SpecialTokenIds::default()
        };
        let ids = SpecialTokenIds::from_tokenizer(&tok, fallback).unwrap();
        assert_eq!((ids.cls_id, ids.sep_id, ids.pad_id), (7, 8, 9));
        assert_eq!((ids.ent_id, ids.rel_id, ids.inner_sep_id), (11, 13, 12));
    }

    #[test]
    fn from_tokenizer_errors_listing_missing_markers() {
        // <<ENT>> present, <<REL>> and <<SEP>> missing -> hard error rather than
        // silently substituting wrong-vocab default IDs.
        let tok = tokenizer_with(&[("[CLS]", 1), ("<<ENT>>", 5)]);
        let err = SpecialTokenIds::from_tokenizer(&tok, SpecialTokenIds::default())
            .expect_err("missing required markers must error");
        let msg = err.to_string();
        assert!(msg.contains("<<REL>>"), "should list <<REL>>: {msg}");
        assert!(msg.contains("<<SEP>>"), "should list <<SEP>>: {msg}");
        assert!(
            !msg.contains("<<ENT>>"),
            "present marker should not be listed: {msg}"
        );
    }
}
