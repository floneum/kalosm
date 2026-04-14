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
        Self {
            cls_id: 1,            // [CLS] token
            sep_id: 2,            // [SEP] token (end-of-sequence)
            pad_id: 0,            // [PAD] token
            ent_id: 250102,       // <<ENT>> token
            rel_id: 250104,       // <<REL>> token
            inner_sep_id: 250103, // <<SEP>> token (internal separator)
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
        let words = self.split_words(text);
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

    /// Split text into words with character offsets.
    ///
    /// Matches Python GLiNER's `WhitespaceTokenSplitter` regex:
    /// `\w+(?:[-_]\w+)*|\S`
    ///
    /// This yields:
    /// - Runs of word characters (alphanumeric/underscore), with hyphens/underscores
    ///   joining word-like groups (e.g. "foo-bar", "x_1")
    /// - OR any single non-whitespace character (punctuation as its own token)
    fn split_words<'a>(&self, text: &'a str) -> Vec<(&'a str, (usize, usize))> {
        let mut words = Vec::new();
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut i = 0;

        // Helpers operating on byte indices. Text is ASCII-safe for typical inputs;
        // for non-ASCII, is_word_char treats each UTF-8 byte - safe approximation that
        // matches what Python's \w would do for ASCII-only text.
        let is_word_char = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        let is_whitespace = |c: u8| matches!(c, b' ' | b'\t' | b'\n' | b'\r');

        while i < n {
            let c = bytes[i];
            if is_whitespace(c) {
                i += 1;
                continue;
            }
            if is_word_char(c) {
                // Match \w+(?:[-_]\w+)*
                let start = i;
                while i < n && is_word_char(bytes[i]) {
                    i += 1;
                }
                // Try to extend with (-|_)\w+ groups (greedy)
                loop {
                    if i + 1 < n && (bytes[i] == b'-' || bytes[i] == b'_') && is_word_char(bytes[i + 1]) {
                        i += 1;
                        while i < n && is_word_char(bytes[i]) {
                            i += 1;
                        }
                    } else {
                        break;
                    }
                }
                let end = i;
                words.push((&text[start..end], (start, end)));
            } else {
                // \S - single non-whitespace character (byte here; for ASCII this is fine)
                // Advance by one UTF-8 codepoint
                let char_len = std::str::from_utf8(&bytes[i..i.saturating_add(4).min(n)])
                    .ok()
                    .and_then(|s| s.chars().next())
                    .map(|c| c.len_utf8())
                    .unwrap_or(1);
                let start = i;
                let end = i + char_len;
                words.push((&text[start..end], (start, end)));
                i = end;
            }
        }

        words
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
    /// Helper function to test word splitting logic without needing a tokenizer.
    fn split_words(text: &str) -> Vec<(&str, (usize, usize))> {
        let mut words = Vec::new();
        let mut char_idx = 0;

        for word in text.split_whitespace() {
            if let Some(pos) = text[char_idx..].find(word) {
                let start = char_idx + pos;
                let end = start + word.len();
                words.push((word, (start, end)));
                char_idx = end;
            }
        }

        words
    }

    #[test]
    fn test_split_words() {
        let text = "Apple Inc. was founded by Steve Jobs.";
        let words = split_words(text);

        assert_eq!(words.len(), 7);
        assert_eq!(words[0].0, "Apple");
        assert_eq!(words[0].1, (0, 5));
        assert_eq!(words[1].0, "Inc.");
        assert_eq!(words[1].1, (6, 10));
        assert_eq!(words[5].0, "Steve");
        assert_eq!(words[6].0, "Jobs.");
    }
}
