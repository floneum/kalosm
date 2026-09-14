//! The training text: a slice of TinyStories, embedded in the binary.
//!
//! Embedded rather than fetched, for the same reason the doodle demo embedded
//! MNIST: a demo that needs a working network and a cooperative CORS policy
//! before it can show anything fails in front of people.
//!
//! TinyStories is short children's stories written with a small vocabulary,
//! which is what makes a quarter-million-parameter model worth watching: the
//! grammar it has to learn is simple enough to actually learn in a minute.

/// Fraction of the text held out. The tail, so a held-out batch can never
/// overlap a training batch.
const TEST_SHARE: f32 = 0.1;

static TEXT: &str = include_str!("../../assets/tinystories.txt");

/// The corpus as token ids, plus the character each id denotes.
pub struct Corpus {
    /// Every character of the text, as a vocabulary index.
    tokens: Vec<u8>,
    /// Index -> character.
    vocab: Vec<char>,
    /// Character -> index, over the vocabulary's contiguous code point range.
    index: Vec<Option<u8>>,
    lowest: u32,
    /// Where the held-out tail starts.
    split: usize,
}

impl Corpus {
    /// Tokenize the embedded text.
    ///
    /// The vocabulary is the set of characters that actually occur, sorted —
    /// derived from the text rather than declared beside it, so the two can
    /// never disagree.
    pub fn load() -> Self {
        let mut seen = [false; 128];
        for c in TEXT.chars() {
            seen[c as usize & 0x7f] = true;
        }
        let vocab: Vec<char> = (0u32..128)
            .filter(|c| seen[*c as usize])
            .filter_map(char::from_u32)
            .collect();
        let lowest = vocab.first().map_or(0, |c| *c as u32);
        let highest = vocab.last().map_or(0, |c| *c as u32);
        let mut index = vec![None; (highest - lowest + 1) as usize];
        for (i, c) in vocab.iter().enumerate() {
            index[(*c as u32 - lowest) as usize] = Some(i as u8);
        }
        let tokens: Vec<u8> = TEXT
            .chars()
            .filter_map(|c| index.get((c as u32).wrapping_sub(lowest) as usize).copied())
            .flatten()
            .collect();
        let split = ((tokens.len() as f32) * (1.0 - TEST_SHARE)) as usize;
        Self {
            tokens,
            vocab,
            index,
            lowest,
            split,
        }
    }

    /// How many distinct characters the model has to spell with.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// The character an id denotes.
    pub fn decode(&self, id: usize) -> char {
        self.vocab.get(id).copied().unwrap_or('?')
    }

    /// Every character, in id order.
    pub fn alphabet(&self) -> &[char] {
        &self.vocab
    }

    /// The id of `c`, when the corpus contains it.
    pub fn encode(&self, c: char) -> Option<u8> {
        self.index
            .get((c as u32).wrapping_sub(self.lowest) as usize)
            .copied()
            .flatten()
    }

    /// `text` as ids, dropping characters the corpus never used.
    pub fn encode_all(&self, text: &str) -> Vec<u8> {
        text.chars().filter_map(|c| self.encode(c)).collect()
    }

    /// Total characters.
    pub fn len(&self) -> usize {
        self.tokens.len()
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

    /// A readable excerpt of the training text, for the UI to show what the
    /// model is being asked to imitate.
    pub fn excerpt(&self, chars: usize) -> String {
        TEXT.chars().take(chars).collect()
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
