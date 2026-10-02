//! A small byte-pair-encoding tokenizer, learned from the training text.
//!
//! Text is pre-split into chunks (an optional leading space plus a run of
//! letters or digits, or any single other character) and merges never cross
//! a chunk. Training repeatedly merges the most frequent adjacent pair over
//! the distinct chunks, weighted by how often each occurs, until the
//! vocabulary is full. Ties break on the smaller pair of ids, so every device
//! learns the same vocabulary from the same text. Ids fit a `u8`.
use std::collections::HashMap;

/// A BPE vocabulary of at most 256 tokens, so every id fits a `u8`.
pub const MAX_VOCAB: usize = 256;

#[derive(PartialEq)]
pub struct Tokenizer {
    /// Id -> the text it stands for.
    pieces: Vec<String>,
    /// Character -> base id.
    base: [Option<u8>; 128],
    /// `(left, right)` -> `(rank, merged id)`; lower ranks merge first.
    merges: HashMap<(u8, u8), (usize, u8)>,
}

impl Tokenizer {
    /// Every character of `alphabet` as a base token, then merges learned
    /// only from `merges_from` until `vocab` tokens (at most [`MAX_VOCAB`]),
    /// so held-out text can widen the alphabet without shaping the merges.
    /// A `vocab` at or below the character count is character-level.
    pub fn train_on(alphabet: &str, merges_from: &str, vocab: usize) -> Self {
        assert!(
            alphabet.is_ascii() && merges_from.is_ascii(),
            "the tokenizer works on ASCII text"
        );
        let text = merges_from;
        let vocab = vocab.min(MAX_VOCAB);
        let mut seen = [false; 128];
        for c in alphabet.bytes() {
            seen[c as usize] = true;
        }
        let mut base = [None; 128];
        let mut pieces = Vec::new();
        for c in 0..128u8 {
            if seen[c as usize] {
                base[c as usize] = Some(pieces.len() as u8);
                pieces.push((c as char).to_string());
            }
        }
        let mut counts: HashMap<&str, u64> = HashMap::new();
        for chunk in chunks(text) {
            *counts.entry(chunk).or_default() += 1;
        }
        let mut words: Vec<(Vec<u8>, u64)> = counts
            .into_iter()
            .map(|(w, n)| (w.bytes().map(|b| base[b as usize].unwrap()).collect(), n))
            .collect();
        let mut merges = HashMap::new();
        while pieces.len() < vocab {
            let mut pairs: HashMap<(u8, u8), u64> = HashMap::new();
            for (ids, n) in &words {
                for pair in ids.windows(2) {
                    *pairs.entry((pair[0], pair[1])).or_default() += n;
                }
            }
            let Some((&pair, _)) = pairs
                .iter()
                .max_by(|(pa, ca), (pb, cb)| ca.cmp(cb).then(pb.cmp(pa)))
            else {
                break;
            };
            let id = pieces.len() as u8;
            pieces.push(format!(
                "{}{}",
                pieces[pair.0 as usize], pieces[pair.1 as usize]
            ));
            merges.insert(pair, (merges.len(), id));
            for (ids, _) in &mut words {
                merge_pair(ids, pair, id);
            }
        }
        Self {
            pieces,
            base,
            merges,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// The text an id stands for.
    pub fn decode(&self, id: usize) -> &str {
        self.pieces.get(id).map_or("?", String::as_str)
    }

    /// Every token's text, in id order.
    pub fn pieces(&self) -> &[String] {
        &self.pieces
    }

    /// `text` as ids, dropping characters the vocabulary does not know.
    pub fn encode(&self, text: &str) -> Vec<u8> {
        let mut cache: HashMap<&str, Vec<u8>> = HashMap::new();
        let mut out = Vec::with_capacity(text.len() / 2);
        for chunk in chunks(text) {
            let ids = cache
                .entry(chunk)
                .or_insert_with(|| self.encode_chunk(chunk));
            out.extend_from_slice(ids);
        }
        out
    }

    fn encode_chunk(&self, chunk: &str) -> Vec<u8> {
        let mut ids: Vec<u8> = chunk
            .bytes()
            .filter_map(|b| self.base.get(b as usize).copied().flatten())
            .collect();
        // Apply merges in the order they were learned.
        loop {
            let best = ids
                .windows(2)
                .filter_map(|p| self.merges.get(&(p[0], p[1])).map(|m| ((p[0], p[1]), *m)))
                .min_by_key(|(_, (rank, _))| *rank);
            let Some((pair, (_, id))) = best else {
                return ids;
            };
            merge_pair(&mut ids, pair, id);
        }
    }
}

/// Replace every non-overlapping `pair` in `ids`, left to right, with `id`.
fn merge_pair(ids: &mut Vec<u8>, pair: (u8, u8), id: u8) {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            out.push(id);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    *ids = out;
}

/// Pre-tokenization: a space may lead a run of letters or digits; every
/// other character stands alone.
fn chunks(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    let mut at = 0;
    std::iter::from_fn(move || {
        if at >= bytes.len() {
            return None;
        }
        let start = at;
        let word = |b: u8| b.is_ascii_alphanumeric();
        if bytes[at] == b' ' && bytes.get(at + 1).is_some_and(|b| word(*b)) {
            at += 1;
        }
        if word(bytes[at]) {
            while at < bytes.len() && word(bytes[at]) {
                at += 1;
            }
        } else {
            at += 1;
        }
        Some(&text[start..at])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "Once upon a time, there was a little girl named Lily. \
        She liked to play. One day, Lily saw a little dog.\n\nThe dog was happy.";

    #[test]
    fn encoding_round_trips_and_compresses() {
        let tokenizer = Tokenizer::train_on(TEXT, TEXT, 80);
        let ids = tokenizer.encode(TEXT);
        let text: String = ids
            .iter()
            .map(|id| tokenizer.decode(*id as usize))
            .collect();
        assert_eq!(text, TEXT);
        assert!(
            ids.len() < TEXT.len() * 3 / 4,
            "{} ids for {} chars",
            ids.len(),
            TEXT.len()
        );
        assert!(tokenizer.vocab_size() <= 80);
    }

    #[test]
    fn training_is_deterministic() {
        let a = Tokenizer::train_on(TEXT, TEXT, 90);
        let b = Tokenizer::train_on(TEXT, TEXT, 90);
        assert_eq!(a.pieces(), b.pieces());
    }

    #[test]
    fn a_vocabulary_no_larger_than_the_alphabet_is_character_level() {
        let tokenizer = Tokenizer::train_on(TEXT, TEXT, 0);
        assert!(tokenizer.pieces().iter().all(|p| p.len() == 1));
        assert_eq!(tokenizer.encode("Lily").len(), 4);
    }

    #[test]
    fn merges_stay_inside_chunks_and_unknown_characters_drop() {
        let tokenizer = Tokenizer::train_on(TEXT, TEXT, 120);
        assert!(tokenizer.pieces().iter().all(|p| {
            p.len() == 1
                || p.trim_start_matches(' ')
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric())
        }));
        let ids = tokenizer.encode("Lily~");
        let text: String = ids
            .iter()
            .map(|id| tokenizer.decode(*id as usize))
            .collect();
        assert_eq!(text, "Lily");
    }
}
