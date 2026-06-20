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
        let words: Vec<String> = split_words
            .iter()
            .map(|(word, _)| word.to_string())
            .collect();
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
pub fn first_subtoken_pooling(
    token_embeddings: &Tensor<3, f32>,
    tokenized: &[TokenizedText],
    device: &Device,
) -> Tensor<3, f32> {
    let positions: Vec<Vec<usize>> = tokenized
        .iter()
        .map(|t| t.word_first_token[..t.num_words].to_vec())
        .collect();
    gather_positions(token_embeddings, &positions, device)
}

/// Enumerate every `(start_word, end_word)` span (inclusive) up to `max_width`
/// words wide within a `num_words`-word sequence.
pub(crate) fn enumerate_spans(num_words: usize, max_width: usize) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    for start in 0..num_words {
        let width_limit = max_width.min(num_words - start);
        for width in 1..=width_limit {
            spans.push((start, start + width - 1));
        }
    }
    spans
}

/// Pad a batch of `(token_ids, attention_mask)` sequences to the longest length
/// and stack them into `[batch, max_seq_len]` tensors.
pub(crate) fn pad_and_stack_inputs(
    device: &Device,
    sequences: &[(&[u32], &[u32])],
    pad_id: u32,
) -> (Tensor<2, u32>, Tensor<2, u32>) {
    let batch_size = sequences.len();
    let max_seq_len = sequences
        .iter()
        .map(|(ids, _)| ids.len())
        .max()
        .unwrap_or(1)
        .max(1);

    let mut token_ids = vec![pad_id; batch_size * max_seq_len];
    let mut attention_mask = vec![0u32; batch_size * max_seq_len];
    for (batch_idx, (ids, mask)) in sequences.iter().enumerate() {
        let offset = batch_idx * max_seq_len;
        let len = ids.len();
        token_ids[offset..offset + len].copy_from_slice(ids);
        attention_mask[offset..offset + len].copy_from_slice(mask);
    }

    let token_ids = Tensor::new(device, &token_ids)
        .reshape([batch_size, max_seq_len])
        .to_concrete();
    let attention_mask = Tensor::new(device, &attention_mask)
        .reshape([batch_size, max_seq_len])
        .to_concrete();
    (token_ids, attention_mask)
}

/// Gather hidden states at the given positions for each batch item, padding
/// shorter position lists with index 0. Returns `[batch, max_positions, hidden]`.
pub(crate) fn gather_positions(
    hidden_states: &Tensor<3, f32>,
    positions_per_batch: &[Vec<usize>],
    device: &Device,
) -> Tensor<3, f32> {
    let [batch_size, seq_len, hidden_size] = hidden_states.shape();
    assert_eq!(
        batch_size,
        positions_per_batch.len(),
        "positions_per_batch must match batch size"
    );

    let max_positions = positions_per_batch.iter().map(Vec::len).max().unwrap_or(0);
    if max_positions == 0 {
        return Tensor::zeros(device, [batch_size, 1, hidden_size]);
    }

    let hidden_flat = hidden_states
        .to_concrete()
        .reshape([batch_size * seq_len, hidden_size])
        .to_concrete();

    // Each gathered position must stay within its row's `seq_len`, otherwise the
    // flat `index_select` would read into a neighbouring sequence. Padded slots
    // use position 0 (masked downstream).
    let mut offset_indices = Vec::with_capacity(batch_size * max_positions);
    for (batch_idx, positions) in positions_per_batch.iter().enumerate() {
        let offset = (batch_idx * seq_len) as u32;
        for pos_idx in 0..max_positions {
            let pos = positions.get(pos_idx).copied().unwrap_or(0);
            debug_assert!(
                pos < seq_len,
                "gather position {pos} out of range for seq_len {seq_len} (batch {batch_idx})"
            );
            offset_indices.push(pos as u32 + offset);
        }
    }

    let offset_indices = Tensor::new(device, &offset_indices);
    hidden_flat
        .index_select(0, &offset_indices)
        .reshape([batch_size, max_positions, hidden_size])
        .to_concrete()
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

        assert!(
            ranges.len() > 1,
            "expected multiple windows, got {ranges:?}"
        );
        // Full coverage: first window starts at the beginning, last reaches the end.
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, text.len());
        // Each window advances and overlaps its predecessor (no gaps, no stalls).
        for pair in ranges.windows(2) {
            assert!(
                pair[1].start > pair[0].start,
                "windows must advance: {ranges:?}"
            );
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
