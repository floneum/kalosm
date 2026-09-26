//! Verified, cached training data, loaded independently of the application.

const SNAPSHOT: &str = "https://raw.githubusercontent.com/floneum/kalosm/117a8a095a5371f5cc78a6b9050ad0eaf89ef314/fusor/webgpu-runner/assets";
const TRAIN_HASH: &str = "4b26bc79469080b97a984fd79d3e61ebd5c3b86f7b9cf79fbc463a349b6cc7e2";
const BENCHMARK_HASH: &str = "87b2de1e357184eee2de1cfa116ac7d41a202cf84ef89068cd86867e905e10e7";

use super::tokenizer::{MAX_VOCAB, Tokenizer};

/// The corpus as token ids, plus the tokenizer that produced them.
#[derive(PartialEq)]
pub struct Corpus {
    /// The whole text, as token ids.
    tokens: Vec<u8>,
    tokenizer: Tokenizer,
    /// Where the held-out tail starts, in tokens.
    split: usize,
}

impl Corpus {
    /// Download once, or reuse the verified local copy on subsequent visits.
    /// Tokens are byte-pair merges learned from the training stories.
    pub async fn load() -> Result<Self, String> {
        let text = fetch("tinystories.txt", 7_999_444, TRAIN_HASH).await?;
        let split = text[..text.len() * 9 / 10]
            .rfind("\n\n\n")
            .ok_or("Training text has no story boundary")?;
        Ok(Self::from_text(&text, split, MAX_VOCAB))
    }

    /// Load the fixed corpus used by compiler benchmarks. It stays
    /// character-level so benchmark timings compare across versions.
    #[allow(dead_code)]
    pub async fn benchmark() -> Result<Self, String> {
        let text = fetch("tinystories-benchmark.txt", 319_868, BENCHMARK_HASH).await?;
        Ok(Self::from_text(
            &text,
            ((text.len() as f32) * 0.9) as usize,
            0,
        ))
    }

    /// `split` is a byte offset; merges are learned from the text before it.
    fn from_text(text: &str, split: usize, vocab: usize) -> Self {
        assert!(
            text.is_ascii(),
            "Corpus preparation must normalize text to ASCII"
        );
        let (train, test) = text.split_at(split);
        let tokenizer = Tokenizer::train_on(text, train, vocab);
        let mut tokens = tokenizer.encode(train);
        let split = tokens.len();
        tokens.extend(tokenizer.encode(test));
        Self {
            tokens,
            tokenizer,
            split,
        }
    }

    /// How many distinct tokens the model has to write with.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// The text a token id stands for.
    pub fn decode(&self, id: usize) -> &str {
        self.tokenizer.decode(id)
    }

    /// Every token's text, in id order.
    pub fn pieces(&self) -> &[String] {
        self.tokenizer.pieces()
    }

    /// `text` as ids, dropping characters the corpus never used.
    pub fn encode_all(&self, text: &str) -> Vec<u8> {
        self.tokenizer.encode(text)
    }

    /// Total tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn split_len(&self, split: Split) -> usize {
        match split {
            Split::Train => self.split,
            Split::Test => self.tokens.len() - self.split,
        }
    }

    /// A `span + 1`-token window starting at `at`, clamped into `range`.
    ///
    /// The extra token is the last position's target: a window of `span`
    /// inputs needs `span` targets, and the target of the final input is the
    /// token after it.
    pub fn window(&self, range: Split, at: usize, span: usize) -> &[u8] {
        let (lo, hi) = match range {
            Split::Train => (0, self.split),
            Split::Test => (self.split, self.tokens.len()),
        };
        let last = hi.saturating_sub(span + 1).max(lo);
        let start = lo + at % (last - lo).max(1);
        &self.tokens[start..(start + span + 1).min(hi)]
    }

    /// A readable excerpt of the training text, `tokens` tokens long, for
    /// the UI to show what the model is being asked to imitate.
    pub fn excerpt(&self, tokens: usize) -> String {
        self.tokens
            .iter()
            .take(tokens)
            .map(|id| self.decode(*id as usize))
            .collect()
    }
}

#[cfg(target_arch = "wasm32")]
async fn fetch(file: &str, size: usize, hash: &str) -> Result<String, String> {
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen(module = "/src/lm/corpus-cache.js")]
    extern "C" {
        #[wasm_bindgen(catch, js_name = loadCorpus)]
        async fn load_corpus(url: &str, size: u32, hash: &str) -> Result<JsValue, JsValue>;
    }
    load_corpus(&format!("{SNAPSHOT}/{file}"), size as u32, hash)
        .await
        .map_err(|error| {
            error
                .as_string()
                .unwrap_or_else(|| "Corpus download failed".into())
        })?
        .as_string()
        .ok_or_else(|| "Corpus response was not text".into())
}

#[cfg(not(target_arch = "wasm32"))]
fn verified(bytes: &[u8], size: usize, hash: &str) -> bool {
    use sha2::{Digest, Sha256};
    bytes.len() == size && format!("{:x}", Sha256::digest(bytes)) == hash
}

#[cfg(not(target_arch = "wasm32"))]
async fn fetch(file: &str, size: usize, hash: &str) -> Result<String, String> {
    use std::{fs, process::Command};
    let cache = std::env::var_os("FUSOR_CORPUS_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("fusor-corpus-v1"));
    let path = cache.join(format!("{hash}.txt"));
    if let Ok(bytes) = fs::read(&path)
        && verified(&bytes, size, hash)
    {
        return String::from_utf8(bytes).map_err(|e| e.to_string());
    }
    let output = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--connect-timeout",
            "15",
            "--max-time",
            "120",
            "--max-filesize",
            &size.to_string(),
            &format!("{SNAPSHOT}/{file}"),
        ])
        .output()
        .map_err(|e| format!("Could not download corpus with curl: {e}"))?;
    if !output.status.success() || !verified(&output.stdout, size, hash) {
        return Err(format!(
            "Corpus download failed or had incorrect contents: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    if fs::create_dir_all(&cache).is_ok() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temporary = cache.join(format!("{hash}-{}-{nonce}.tmp", std::process::id()));
        if fs::write(&temporary, &text).is_ok() {
            let _ = fs::rename(&temporary, &path);
        }
        let _ = fs::remove_file(&temporary);
    }
    Ok(text)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn expanded_data_round_trips_and_windows_stay_in_their_split() {
        let corpus = pollster::block_on(Corpus::load()).unwrap();
        assert_eq!(corpus.vocab_size(), MAX_VOCAB);
        let text = corpus.excerpt(corpus.len());
        assert_eq!(text.len(), 7_999_444);
        assert!(verified(text.as_bytes(), 7_999_444, TRAIN_HASH));
        // Merges shorten the stories well below one token per character.
        assert!(corpus.len() < text.len() / 2, "{} tokens", corpus.len());
        assert!(corpus.excerpt(corpus.split).len() == 7_199_486);
        for split in [Split::Train, Split::Test] {
            for at in [0, corpus.len() - 1, usize::MAX] {
                let window = corpus.window(split, at, 512);
                assert_eq!(window.len(), 513);
                let offset = window.as_ptr() as usize - corpus.tokens.as_ptr() as usize;
                match split {
                    Split::Train => assert!(offset + window.len() <= corpus.split),
                    Split::Test => assert!(offset >= corpus.split),
                }
            }
        }
    }

    #[test]
    fn the_benchmark_corpus_stays_character_level() {
        let corpus = pollster::block_on(Corpus::benchmark()).unwrap();
        assert!(corpus.pieces().iter().all(|p| p.len() == 1));
        assert_eq!(corpus.len(), 319_868);
    }
}

/// Which half of the corpus a batch is drawn from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Split {
    /// The first 90%, which the optimizer sees.
    Train,
    /// The last 10%, which it never does.
    Test,
}
