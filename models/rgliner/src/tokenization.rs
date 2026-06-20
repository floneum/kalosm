//! Word-level tokenization with subtoken-to-word mapping.

use fusor::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::error::GlinerError;

/// Tokenization result with word-level alignment.
#[derive(Debug, Clone)]
pub struct TokenizedText {
    /// Token IDs for the model.
    pub token_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<u32>,
    /// Index of the first token for each word.
    pub word_first_token: Vec<usize>,
    /// Number of words in the input.
    pub num_words: usize,
    /// Character offsets for each word: (start_char, end_char).
    pub word_offsets: Vec<(usize, usize)>,
}

/// Word-level tokenizer wrapper.
pub struct WordTokenizer {
    pub(crate) tokenizer: Tokenizer,
    add_special_tokens: bool,
}

impl WordTokenizer {
    /// Create a tokenizer.
    ///
    /// `add_special_tokens` controls whether the tokenizer's post-processor is
    /// applied. Set to `false` for encoders whose Python counterpart strips
    /// [CLS]/[SEP] from the post-processor (ModernBERT/ettin have
    /// `add_bos_token=False` because they lack bos/eos tokens — see GLiNER's
    /// `_set_tokenizer_spec_tokens`).
    pub fn new(tokenizer: Tokenizer, add_special_tokens: bool) -> Self {
        Self {
            tokenizer,
            add_special_tokens,
        }
    }

    /// Tokenize text and track word boundaries.
    pub fn tokenize(&self, text: &str) -> Result<TokenizedText, GlinerError> {
        let split_words = split_words(text);
        let words: Vec<String> = split_words.iter().map(|(word, _)| word.to_string()).collect();
        let word_offsets: Vec<(usize, usize)> =
            split_words.iter().map(|(_, offsets)| *offsets).collect();

        let encoding = self
            .tokenizer
            .encode(words, self.add_special_tokens)
            .map_err(GlinerError::Tokenizer)?;

        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as u32)
            .collect();

        // Build token-to-word mapping to find the first token for each word.
        let num_words = word_offsets.len();
        let mut word_first_token = vec![0usize; num_words];
        let mut seen_words = vec![false; num_words];
        for (token_idx, opt) in encoding.get_word_ids().iter().enumerate() {
            if let Some(word_id) = *opt {
                let word_id = word_id as usize;
                if !seen_words[word_id] {
                    word_first_token[word_id] = token_idx;
                    seen_words[word_id] = true;
                }
            }
        }

        Ok(TokenizedText {
            token_ids,
            attention_mask,
            word_first_token,
            num_words,
            word_offsets,
        })
    }

    /// Tokenize a batch of texts.
    pub fn tokenize_batch(&self, texts: &[&str]) -> Result<Vec<TokenizedText>, GlinerError> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }

    /// Resolve the tokenizer's padding ID.
    ///
    /// Prefer the tokenizer's configured padding params, then the conventional
    /// pad token names (`[PAD]` for BERT/DeBERTa, `<pad>` for others), falling
    /// back to `0` only as a last resort. Hardcoding `[PAD]` would pick the wrong
    /// id for encoders (e.g. some ettin/ModernBERT vocabs) that name it `<pad>`.
    pub fn pad_id(&self) -> u32 {
        if let Some(padding) = self.tokenizer.get_padding() {
            return padding.pad_id;
        }
        self.tokenizer
            .token_to_id("[PAD]")
            .or_else(|| self.tokenizer.token_to_id("<pad>"))
            .unwrap_or(0)
    }
}

/// Pack `text` into token-budgeted byte ranges using the supplied tokenizer.
///
/// Splits the input on GLiNER-style word boundaries, encodes each word to count
/// subtokens, and greedily fills windows of at most `token_budget` subtokens with
/// `overlap_tokens` of trailing-token overlap between adjacent windows.
pub(crate) fn token_packed_ranges(
    tokenizer: &Tokenizer,
    text: &str,
    token_budget: usize,
    overlap_tokens: usize,
) -> Result<Vec<std::ops::Range<usize>>, GlinerError> {
    let words = split_words(text);
    if words.is_empty() {
        return Ok(Vec::new());
    }

    let mut word_token_counts = Vec::with_capacity(words.len());
    for (w, _) in &words {
        let enc = tokenizer
            .encode(*w, false)
            .map_err(GlinerError::Tokenizer)?;
        word_token_counts.push(enc.get_ids().len().max(1));
    }

    let mut ranges = Vec::new();
    let mut word = 0usize;
    while word < words.len() {
        let mut end_word = word;
        let mut tokens = 0usize;
        while end_word < words.len() && tokens + word_token_counts[end_word] <= token_budget {
            tokens += word_token_counts[end_word];
            end_word += 1;
        }
        if end_word == word {
            end_word = word + 1;
        }
        ranges.push(words[word].1 .0..words[end_word - 1].1 .1);
        if end_word == words.len() {
            break;
        }
        let mut back_tokens = 0usize;
        let mut next = end_word;
        while next > word + 1 && back_tokens < overlap_tokens {
            next -= 1;
            back_tokens += word_token_counts[next];
        }
        word = next.max(word + 1);
    }
    Ok(ranges)
}

/// Split text into words with byte offsets, matching Python GLiNER's
/// `WhitespaceTokenSplitter` regex `\w+(?:[-_]\w+)*|\S`. `\w` is Unicode-aware
/// (`char::is_alphanumeric`), so this is correct for the multilingual models.
/// Shared by both the bi-encoder and RelEx tokenizers.
pub(crate) fn split_words(text: &str) -> Vec<(&str, (usize, usize))> {
    let mut words = Vec::new();
    let mut chars = text.char_indices().peekable();

    while let Some((start, ch)) = chars.peek().copied() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }

        if is_word_char(ch) {
            chars.next();
            let mut end = start + ch.len_utf8();

            while let Some((idx, next_ch)) = chars.peek().copied() {
                if is_word_char(next_ch) {
                    end = idx + next_ch.len_utf8();
                    chars.next();
                    continue;
                }

                if matches!(next_ch, '-' | '_') {
                    let mut lookahead = chars.clone();
                    lookahead.next();
                    if let Some((_, after_delimiter)) = lookahead.peek().copied() {
                        if is_word_char(after_delimiter) {
                            chars.next();
                            end = idx + next_ch.len_utf8();

                            while let Some((word_idx, word_ch)) = chars.peek().copied() {
                                if !is_word_char(word_ch) {
                                    break;
                                }
                                end = word_idx + word_ch.len_utf8();
                                chars.next();
                            }
                            continue;
                        }
                    }
                }

                break;
            }

            words.push((&text[start..end], (start, end)));
            continue;
        }

        chars.next();
        let end = start + ch.len_utf8();
        words.push((&text[start..end], (start, end)));
    }

    words
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// Pool token embeddings to word embeddings using first-subtoken strategy.
///
/// # Arguments
/// * `token_embeddings` - Token embeddings [batch, seq_len, hidden_dim]
/// * `tokenized` - Tokenization results for each batch item
/// * `device` - Device to create output tensor on
///
/// # Returns
/// * Word embeddings [batch, max_words, hidden_dim]
/// * Word mask [batch, max_words] - 1 for valid words, 0 for padding
pub fn first_subtoken_pooling(
    token_embeddings: &Tensor<3, f32>,
    tokenized: &[TokenizedText],
    device: &Device,
) -> (Tensor<3, f32>, Tensor<2, u32>) {
    let shape = token_embeddings.shape();
    let batch_size = shape[0];
    let hidden_dim = shape[2];

    // Find max words across batch
    let max_words = tokenized.iter().map(|t| t.num_words).max().unwrap_or(0);

    if max_words == 0 {
        // Return empty tensors if no words
        let word_emb = Tensor::zeros(device, [batch_size, 1, hidden_dim]);
        let word_mask = Tensor::zeros(device, [batch_size, 1]);
        return (word_emb, word_mask);
    }

    // Build gather indices for each batch item
    // For each batch, we need to gather word_first_token[w] for each word w
    let mut all_indices: Vec<u32> = Vec::with_capacity(batch_size * max_words);
    let mut mask_data: Vec<u32> = Vec::with_capacity(batch_size * max_words);

    for t in tokenized {
        for word_idx in 0..max_words {
            if word_idx < t.num_words {
                all_indices.push(t.word_first_token[word_idx] as u32);
                mask_data.push(1);
            } else {
                // Padding - use index 0 (will be masked out)
                all_indices.push(0);
                mask_data.push(0);
            }
        }
    }

    // Reshape token_embeddings to [batch_size * seq_len, hidden_dim] for gathering
    let seq_len = shape[1];
    let token_embeddings_concrete = token_embeddings.to_concrete();
    let flat_embeddings = token_embeddings_concrete
        .reshape([batch_size * seq_len, hidden_dim])
        .to_concrete();

    // Offset each word's first-token index into the flattened [batch*seq, hidden]
    // tensor. Every first-token index must land within its row's `seq_len`;
    // otherwise the gather would read into a neighbouring sequence's tokens.
    let mut offset_indices: Vec<u32> = Vec::with_capacity(batch_size * max_words);
    for batch_idx in 0..batch_size {
        let offset = (batch_idx * seq_len) as u32;
        for word_idx in 0..max_words {
            let idx = all_indices[batch_idx * max_words + word_idx];
            debug_assert!(
                (idx as usize) < seq_len,
                "first-token index {idx} out of range for seq_len {seq_len} (batch {batch_idx}, word {word_idx})"
            );
            offset_indices.push(idx + offset);
        }
    }
    let offset_indices_tensor = Tensor::new(device, &offset_indices);

    // Gather word embeddings
    let gathered = flat_embeddings.index_select(0, &offset_indices_tensor);

    // Reshape to [batch_size, max_words, hidden_dim]
    let word_embeddings = gathered
        .reshape([batch_size, max_words, hidden_dim])
        .to_concrete();

    // Create word mask
    let word_mask = Tensor::new(device, &mask_data)
        .reshape([batch_size, max_words])
        .to_concrete();

    (word_embeddings, word_mask)
}

#[cfg(test)]
mod tests {
    use super::{split_words, token_packed_ranges};
    use tokenizers::Tokenizer;

    /// A WordLevel tokenizer where every word encodes to exactly one token (its
    /// id, or `[UNK]`). That makes per-word token counts equal 1, which is
    /// enough to exercise the windowing math deterministically.
    fn simple_tokenizer() -> Tokenizer {
        let json = r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("valid tokenizer json")
    }

    #[test]
    fn token_packed_ranges_cover_input_with_overlap() {
        let tok = simple_tokenizer();
        // 10 one-token words; budget 4 forces multiple overlapping windows.
        let text = "a b c d e f g h i j";
        let ranges = token_packed_ranges(&tok, text, 4, 1).unwrap();

        assert!(ranges.len() > 1, "expected multiple windows, got {ranges:?}");
        // Full coverage: first window starts at the beginning, last reaches the end.
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, text.len());
        // Each window advances and overlaps its predecessor (no gaps, no stalls).
        for pair in ranges.windows(2) {
            assert!(pair[1].start > pair[0].start, "windows must advance: {ranges:?}");
            assert!(
                pair[1].start < pair[0].end,
                "adjacent windows must overlap: {ranges:?}"
            );
        }
    }

    #[test]
    fn token_packed_ranges_single_window_within_budget() {
        let tok = simple_tokenizer();
        let text = "a b c";
        let ranges = token_packed_ranges(&tok, text, 16, 2).unwrap();
        assert_eq!(ranges, vec![0..text.len()]);
    }

    #[test]
    fn token_packed_ranges_empty_text() {
        let tok = simple_tokenizer();
        assert!(token_packed_ranges(&tok, "   ", 8, 1).unwrap().is_empty());
    }

    #[test]
    fn split_words_matches_gliner_word_regex() {
        let words = split_words("all-MiniLM_L6-v2 rocks.");

        assert_eq!(
            words,
            vec![
                ("all-MiniLM_L6-v2", (0, 16)),
                ("rocks", (17, 22)),
                (".", (22, 23)),
            ]
        );
    }

    #[test]
    fn split_words_keeps_punctuation_as_separate_words() {
        let words = split_words("Apple Inc. was founded.");

        assert_eq!(
            words,
            vec![
                ("Apple", (0, 5)),
                ("Inc", (6, 9)),
                (".", (9, 10)),
                ("was", (11, 14)),
                ("founded", (15, 22)),
                (".", (22, 23)),
            ]
        );
    }
}
