//! Minimal byte-level BPE tokenization used by Kalosm GGUF models.

use rustc_hash::{FxHashMap, FxHashSet};
use thiserror::Error;

const MISSING_BYTE_TOKEN: u32 = u32::MAX;
const GREEDY_MAX_INPUT_BYTES: usize = 128;
const NO_CANDIDATE_LEVEL: u16 = u16::MAX;
const MERGE_TABLE_BUCKETS: usize = 256 * 256;
const CANDIDATE_LOOKUP_CACHE_BITS: u32 = 4;
const CANDIDATE_LOOKUP_CACHE_SLOTS: usize = 1 << CANDIDATE_LOOKUP_CACHE_BITS;

type PairKey = u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ByteTokenMapping {
    Table,
    Identity,
    Gpt2,
}

fn classify_byte_token_mapping(byte_to_token: &[u32; 256]) -> ByteTokenMapping {
    if byte_to_token
        .iter()
        .copied()
        .enumerate()
        .all(|(byte, token)| token == byte as u32)
    {
        ByteTokenMapping::Identity
    } else if byte_to_token
        .iter()
        .copied()
        .enumerate()
        .all(|(byte, token)| token == gpt2_byte_token(byte as u8))
    {
        ByteTokenMapping::Gpt2
    } else {
        ByteTokenMapping::Table
    }
}

#[inline(always)]
fn gpt2_byte_token(byte: u8) -> u32 {
    match byte {
        33..=126 => byte as u32 - 33,
        161..=172 => byte as u32 - 67,
        174..=255 => byte as u32 - 68,
        173 => 255,
        127..=160 => byte as u32 + 94,
        0..=32 => byte as u32 + 188,
    }
}

/// A byte-level BPE tokenizer.
#[derive(Clone, Debug)]
pub struct FastBpe {
    byte_to_token: [u32; 256],
    all_bytes_present: bool,
    byte_token_mapping: ByteTokenMapping,
    token_bytes: FxHashMap<Vec<u8>, u32>,
    id_to_bytes: Vec<Option<Vec<u8>>>,
    merges: FxHashMap<PairKey, MergeRule>,
    level_merges: MergeLookup,
    levels: Vec<MergeLevel>,
    ignore_merges: bool,
}

impl FastBpe {
    /// Build a byte-level BPE tokenizer from vocab and merge rules.
    pub fn from_vocab_and_merges(
        vocab: impl IntoIterator<Item = (String, u32)>,
        merges: impl IntoIterator<Item = String>,
        ignore_merges: bool,
    ) -> Result<Self, TokenizerError> {
        let mut byte_to_token = [MISSING_BYTE_TOKEN; 256];
        let mut token_bytes = FxHashMap::default();
        let mut id_to_bytes = Vec::new();

        for (token, id) in vocab {
            let bytes = decode_token_bytes(&token);
            if bytes.len() == 1 {
                if id == MISSING_BYTE_TOKEN {
                    return Err(TokenizerError::ReservedByteTokenId(id));
                }
                byte_to_token[bytes[0] as usize] = id;
            }
            let id = id as usize;
            if id >= id_to_bytes.len() {
                id_to_bytes.resize(id + 1, None);
            }
            id_to_bytes[id] = Some(bytes.clone());
            token_bytes.insert(bytes, id as u32);
        }

        let all_bytes_present = byte_to_token
            .iter()
            .all(|token| *token != MISSING_BYTE_TOKEN);
        let byte_token_mapping = classify_byte_token_mapping(&byte_to_token);

        let raw_merges = merges
            .into_iter()
            .enumerate()
            .map(|(rank, merge)| parse_merge(rank as u32, &merge, &token_bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let merge_levels = assign_merge_levels(&raw_merges);
        let level_count = merge_levels
            .iter()
            .copied()
            .max()
            .map_or(0, |level| level as usize + 1);
        let mut merges_by_level = vec![Vec::new(); level_count];
        let mut merges_by_pair = FxHashMap::default();

        for (merge, level) in raw_merges.iter().copied().zip(merge_levels.iter().copied()) {
            let pair = pair_key(merge.pair.left, merge.pair.right);
            let rule = MergeRule {
                rank: merge.rank,
                new_token: merge.new_token,
            };
            merges_by_pair.insert(pair, rule);
            merges_by_level[level as usize].push((pair, rule));
        }

        let level_merges = MergeLookup::new(
            raw_merges
                .iter()
                .copied()
                .zip(merge_levels.iter().copied())
                .map(|(merge, level)| MergeEntry {
                    key: pair_key(merge.pair.left, merge.pair.right),
                    level,
                    new_token: merge.new_token,
                }),
        );

        let levels = merges_by_level
            .into_iter()
            .map(|mut merges| {
                if merges.len() == 1 {
                    let (pair, rule) = merges.pop().unwrap();
                    let (left, right) = split_pair_key(pair);
                    MergeLevel::Single {
                        left,
                        right,
                        new_token: rule.new_token,
                    }
                } else {
                    MergeLevel::Multiple
                }
            })
            .collect();

        Ok(Self {
            byte_to_token,
            all_bytes_present,
            byte_token_mapping,
            token_bytes,
            id_to_bytes,
            merges: merges_by_pair,
            level_merges,
            levels,
            ignore_merges,
        })
    }

    /// Tokenize a byte slice.
    pub fn tokenize(&self, input: &[u8]) -> Result<Vec<u32>, TokenizerError> {
        let mut buffers = TokenizationBuffers::default();
        self.tokenize_into(input, &mut buffers)?;
        Ok(buffers.tokens)
    }

    /// Tokenize a batch of independent byte slices, concatenating token ids in input order.
    pub fn tokenize_batch(&self, inputs: &[&[u8]]) -> Result<Vec<u32>, TokenizerError> {
        let mut buffers = BatchTokenizationBuffers::default();
        self.tokenize_batch_into(inputs, &mut buffers)?;
        Ok(buffers.tokens)
    }

    /// Tokenize into reusable buffers.
    pub fn tokenize_into<'a>(
        &self,
        input: &[u8],
        buffers: &'a mut TokenizationBuffers,
    ) -> Result<&'a [u32], TokenizerError> {
        buffers.tokens.clear();

        if self.ignore_merges {
            if let Some(token) = self.token_bytes.get(input) {
                buffers.tokens.push(*token);
                return Ok(&buffers.tokens);
            }
        }

        if self.should_use_greedy(input.len()) {
            self.encode_bytes_into(input, &mut buffers.tokens)?;
            apply_greedy_merges(&self.merges, &mut buffers.tokens);
            return Ok(&buffers.tokens);
        }

        let first_unapplied_level = match self.levels.first() {
            Some(MergeLevel::Single {
                left,
                right,
                new_token,
            }) => {
                self.encode_bytes_and_apply_single_merge(
                    input,
                    &mut buffers.tokens,
                    *left,
                    *right,
                    *new_token,
                )?;
                1
            }
            _ => {
                self.encode_bytes_into(input, &mut buffers.tokens)?;
                0
            }
        };

        let mut next_level = rebuild_merge_candidates(
            &self.level_merges,
            &buffers.tokens,
            &mut buffers.candidate_levels,
            &mut buffers.candidate_new_tokens,
            first_unapplied_level,
        );

        while let Some(level_index) = next_level {
            next_level =
                self.apply_candidate_merge_level(level_index as u16, level_index + 1, buffers);
        }

        Ok(&buffers.tokens)
    }

    /// Tokenize a batch into reusable buffers, concatenating token ids in input order.
    pub fn tokenize_batch_into<'a>(
        &self,
        inputs: &[&[u8]],
        buffers: &'a mut BatchTokenizationBuffers,
    ) -> Result<&'a [u32], TokenizerError> {
        buffers.tokens.clear();
        buffers
            .tokens
            .reserve(inputs.iter().map(|input| input.len()).sum());

        for input in inputs {
            let tokens = self.tokenize_into(input, &mut buffers.scratch)?;
            buffers.tokens.extend_from_slice(tokens);
        }

        Ok(&buffers.tokens)
    }

    /// Reference BPE implementation used by tests.
    pub fn tokenize_reference(&self, input: &[u8]) -> Result<Vec<u32>, TokenizerError> {
        let mut tokens = Vec::new();
        self.encode_bytes_into(input, &mut tokens)?;
        apply_greedy_merges(&self.merges, &mut tokens);
        Ok(tokens)
    }

    /// Get the decoded raw bytes for a token id.
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.id_to_bytes
            .get(id as usize)
            .and_then(Option::as_ref)
            .map(Vec::as_slice)
    }

    fn encode_bytes_into(&self, input: &[u8], out: &mut Vec<u32>) -> Result<(), TokenizerError> {
        out.clear();
        out.resize(input.len(), 0);

        if self.all_bytes_present {
            encode_bytes_unchecked(input, &self.byte_to_token, self.byte_token_mapping, out);
        } else {
            encode_bytes_checked(input, &self.byte_to_token, out)?;
        }

        Ok(())
    }

    fn encode_bytes_and_apply_single_merge(
        &self,
        input: &[u8],
        out: &mut Vec<u32>,
        left: u32,
        right: u32,
        new_token: u32,
    ) -> Result<(), TokenizerError> {
        if self.all_bytes_present {
            self.encode_bytes_and_apply_single_merge_unchecked(input, out, left, right, new_token);
            Ok(())
        } else {
            self.encode_bytes_and_apply_single_merge_checked(input, out, left, right, new_token)
        }
    }

    fn encode_bytes_and_apply_single_merge_checked(
        &self,
        input: &[u8],
        out: &mut Vec<u32>,
        left: u32,
        right: u32,
        new_token: u32,
    ) -> Result<(), TokenizerError> {
        out.clear();
        out.reserve(input.len());

        let mut index = 0;
        while index + 1 < input.len() {
            let token = match lookup_byte_token(&self.byte_to_token, input[index]) {
                Ok(token) => token,
                Err(err) => {
                    out.clear();
                    return Err(err);
                }
            };

            if token == left {
                let next_token = match lookup_byte_token(&self.byte_to_token, input[index + 1]) {
                    Ok(token) => token,
                    Err(err) => {
                        out.clear();
                        return Err(err);
                    }
                };

                if next_token == right {
                    out.push(new_token);
                    index += 2;
                    continue;
                }
            }

            out.push(token);
            index += 1;
        }

        if index < input.len() {
            match lookup_byte_token(&self.byte_to_token, input[index]) {
                Ok(token) => out.push(token),
                Err(err) => {
                    out.clear();
                    return Err(err);
                }
            }
        }

        Ok(())
    }

    fn encode_bytes_and_apply_single_merge_unchecked(
        &self,
        input: &[u8],
        out: &mut Vec<u32>,
        left: u32,
        right: u32,
        new_token: u32,
    ) {
        out.clear();
        out.reserve(input.len());

        let mut index = 0;
        while index + 1 < input.len() {
            let token = self.byte_token_unchecked(input[index]);

            if token == left {
                let next_token = self.byte_token_unchecked(input[index + 1]);
                if next_token == right {
                    out.push(new_token);
                    index += 2;
                    continue;
                }
            }

            out.push(token);
            index += 1;
        }

        if index < input.len() {
            out.push(self.byte_token_unchecked(input[index]));
        }
    }

    #[inline(always)]
    fn byte_token_unchecked(&self, byte: u8) -> u32 {
        match self.byte_token_mapping {
            ByteTokenMapping::Table => self.byte_to_token[byte as usize],
            ByteTokenMapping::Identity => byte as u32,
            ByteTokenMapping::Gpt2 => gpt2_byte_token(byte),
        }
    }

    fn should_use_greedy(&self, input_len: usize) -> bool {
        input_len <= GREEDY_MAX_INPUT_BYTES && self.levels.len() > input_len.saturating_mul(2)
    }

    fn apply_candidate_merge_level(
        &self,
        level: u16,
        min_next_level: usize,
        buffers: &mut TokenizationBuffers,
    ) -> Option<usize> {
        apply_candidate_merge_level(&self.level_merges, level, min_next_level, buffers)
    }
}

/// Reusable buffers for tokenization.
#[derive(Debug, Default)]
pub struct TokenizationBuffers {
    tokens: Vec<u32>,
    next: Vec<u32>,
    candidate_levels: Vec<u16>,
    candidate_new_tokens: Vec<u32>,
    next_candidate_levels: Vec<u16>,
    next_candidate_new_tokens: Vec<u32>,
}

impl TokenizationBuffers {
    /// The tokens from the last tokenization call.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

/// Reusable buffers for batch tokenization.
#[derive(Debug, Default)]
pub struct BatchTokenizationBuffers {
    tokens: Vec<u32>,
    scratch: TokenizationBuffers,
}

impl BatchTokenizationBuffers {
    /// The concatenated tokens from the last batch tokenization call.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

/// Errors from byte-level BPE tokenization.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TokenizerError {
    /// A merge rule did not contain exactly two tokens.
    #[error("merge `{0}` must contain exactly two tokens")]
    InvalidMerge(String),
    /// A token referenced by a merge rule was missing from the vocabulary.
    #[error("token `{0}` is missing from the vocabulary")]
    MissingToken(String),
    /// The reserved missing-byte sentinel was used as a byte-token id.
    #[error("single-byte token id {0} is reserved for missing-byte detection")]
    ReservedByteTokenId(u32),
    /// A byte was missing from the vocabulary.
    #[error("byte 0x{0:02x} is missing from the vocabulary")]
    MissingByteToken(u8),
}

#[inline(always)]
fn encode_bytes_checked(
    input: &[u8],
    byte_to_token: &[u32; 256],
    out: &mut [u32],
) -> Result<(), TokenizerError> {
    for (byte, output) in input.iter().copied().zip(out) {
        *output = lookup_byte_token(byte_to_token, byte)?;
    }
    Ok(())
}

#[inline(always)]
fn encode_bytes_unchecked(
    input: &[u8],
    byte_to_token: &[u32; 256],
    byte_token_mapping: ByteTokenMapping,
    out: &mut [u32],
) {
    match byte_token_mapping {
        ByteTokenMapping::Table => {
            for (byte, output) in input.iter().copied().zip(out) {
                *output = byte_to_token[byte as usize];
            }
        }
        ByteTokenMapping::Identity => encode_identity_bytes(input, out),
        ByteTokenMapping::Gpt2 => encode_gpt2_bytes(input, out),
    }
}

#[inline(always)]
fn encode_gpt2_bytes(input: &[u8], out: &mut [u32]) {
    for (byte, output) in input.iter().copied().zip(out) {
        *output = gpt2_byte_token(byte);
    }
}

#[inline(always)]
fn encode_identity_bytes(input: &[u8], out: &mut [u32]) {
    for (byte, output) in input.iter().copied().zip(out) {
        *output = byte as u32;
    }
}

#[inline(always)]
fn lookup_byte_token(byte_to_token: &[u32; 256], byte: u8) -> Result<u32, TokenizerError> {
    let token = byte_to_token[byte as usize];
    if token == MISSING_BYTE_TOKEN {
        Err(TokenizerError::MissingByteToken(byte))
    } else {
        Ok(token)
    }
}

#[inline(always)]
fn pair_key(left: u32, right: u32) -> PairKey {
    ((left as PairKey) << 32) | right as PairKey
}

#[inline(always)]
fn split_pair_key(pair: PairKey) -> (u32, u32) {
    ((pair >> 32) as u32, pair as u32)
}

#[inline(always)]
fn merge_bucket(pair: PairKey) -> usize {
    let left = (pair >> 32) as usize;
    let right = pair as usize;
    (left & 0xff) | ((right & 0xff) << 8)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TokenPair {
    left: u32,
    right: u32,
}

impl TokenPair {
    fn new(left: u32, right: u32) -> Self {
        Self { left, right }
    }
}

#[derive(Clone, Copy, Debug)]
struct MergeRule {
    rank: u32,
    new_token: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct MergeBucket {
    start: u32,
    end: u32,
}

#[derive(Clone, Copy, Debug)]
struct MergeEntry {
    key: PairKey,
    level: u16,
    new_token: u32,
}

#[derive(Clone, Debug)]
struct MergeLookup {
    buckets: Box<[MergeBucket]>,
    entries: Box<[MergeEntry]>,
}

impl MergeLookup {
    fn new(entries: impl IntoIterator<Item = MergeEntry>) -> Self {
        let mut sorted = entries
            .into_iter()
            .map(|entry| (merge_bucket(entry.key), entry))
            .collect::<Vec<_>>();
        sorted.sort_unstable_by_key(|(bucket, entry)| (*bucket, entry.key));

        let mut buckets = vec![MergeBucket::default(); MERGE_TABLE_BUCKETS];
        let mut compact = Vec::with_capacity(sorted.len());
        let mut index = 0;

        while index < sorted.len() {
            let bucket = sorted[index].0;
            let start = compact.len() as u32;
            while index < sorted.len() && sorted[index].0 == bucket {
                compact.push(sorted[index].1);
                index += 1;
            }
            buckets[bucket] = MergeBucket {
                start,
                end: compact.len() as u32,
            };
        }

        Self {
            buckets: buckets.into_boxed_slice(),
            entries: compact.into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn get_key(&self, key: PairKey) -> Option<MergeEntry> {
        let bucket = self.buckets[merge_bucket(key)];
        self.entries[bucket.start as usize..bucket.end as usize]
            .iter()
            .copied()
            .find(|entry| entry.key == key)
    }
}

#[derive(Clone, Debug)]
enum MergeLevel {
    Single {
        left: u32,
        right: u32,
        new_token: u32,
    },
    Multiple,
}

#[derive(Clone, Copy, Debug)]
struct RawMerge {
    rank: u32,
    pair: TokenPair,
    new_token: u32,
}

fn parse_merge(
    rank: u32,
    merge: &str,
    token_bytes: &FxHashMap<Vec<u8>, u32>,
) -> Result<RawMerge, TokenizerError> {
    let (left, right) = merge
        .split_once(' ')
        .ok_or_else(|| TokenizerError::InvalidMerge(merge.to_owned()))?;
    let left_bytes = decode_token_bytes(left);
    let right_bytes = decode_token_bytes(right);
    let new_bytes = left_bytes
        .iter()
        .chain(right_bytes.iter())
        .copied()
        .collect::<Vec<_>>();

    let left = *token_bytes
        .get(&left_bytes)
        .ok_or_else(|| TokenizerError::MissingToken(left.to_owned()))?;
    let right = *token_bytes
        .get(&right_bytes)
        .ok_or_else(|| TokenizerError::MissingToken(right.to_owned()))?;
    let new_token = *token_bytes
        .get(&new_bytes)
        .ok_or_else(|| TokenizerError::MissingToken(format!("{left} {right}")))?;

    Ok(RawMerge {
        rank,
        pair: TokenPair::new(left, right),
        new_token,
    })
}

fn apply_greedy_merges(merges: &FxHashMap<PairKey, MergeRule>, tokens: &mut Vec<u32>) {
    while let Some((index, new_token)) = best_merge(merges, tokens) {
        tokens[index] = new_token;
        tokens.remove(index + 1);
    }
}

fn rebuild_merge_candidates(
    merges: &MergeLookup,
    tokens: &[u32],
    candidate_levels: &mut Vec<u16>,
    candidate_new_tokens: &mut Vec<u32>,
    min_level: usize,
) -> Option<usize> {
    let pair_count = tokens.len().saturating_sub(1);
    let min_level = u16::try_from(min_level).unwrap_or(NO_CANDIDATE_LEVEL);
    let mut best = NO_CANDIDATE_LEVEL;

    candidate_levels.clear();
    candidate_new_tokens.clear();
    candidate_levels.reserve(pair_count);
    candidate_new_tokens.reserve(pair_count);

    let mut cached_pair = None;
    let mut cached_entry = None;

    for pair in tokens.windows(2) {
        let key = pair_key(pair[0], pair[1]);
        let entry = if cached_pair == Some(key) {
            cached_entry
        } else {
            let entry = merges.get_key(key);
            cached_pair = Some(key);
            cached_entry = entry;
            entry
        };

        if let Some(entry) = entry {
            let level = entry.level;
            if level >= min_level && level < best {
                best = level;
            }
            candidate_levels.push(level);
            candidate_new_tokens.push(entry.new_token);
        } else {
            candidate_levels.push(NO_CANDIDATE_LEVEL);
            candidate_new_tokens.push(0);
        }
    }

    (best != NO_CANDIDATE_LEVEL).then_some(best as usize)
}

#[derive(Clone, Copy)]
struct OutputToken {
    token: u32,
    old_start: usize,
    old_end: usize,
    merged: bool,
}

impl OutputToken {
    fn copied(index: usize, token: u32) -> Self {
        Self {
            token,
            old_start: index,
            old_end: index,
            merged: false,
        }
    }

    fn merged(index: usize, token: u32) -> Self {
        Self {
            token,
            old_start: index,
            old_end: index + 1,
            merged: true,
        }
    }
}

struct CandidateLookupCache {
    key: Option<PairKey>,
    entry: Option<MergeEntry>,
    slots: [CandidateLookupCacheSlot; CANDIDATE_LOOKUP_CACHE_SLOTS],
}

impl Default for CandidateLookupCache {
    fn default() -> Self {
        Self {
            key: None,
            entry: None,
            slots: [CandidateLookupCacheSlot::EMPTY; CANDIDATE_LOOKUP_CACHE_SLOTS],
        }
    }
}

#[derive(Clone, Copy)]
struct CandidateLookupCacheSlot {
    key: PairKey,
    valid: bool,
    entry: Option<MergeEntry>,
}

impl CandidateLookupCacheSlot {
    const EMPTY: Self = Self {
        key: 0,
        valid: false,
        entry: None,
    };
}

impl Default for CandidateLookupCacheSlot {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl CandidateLookupCache {
    #[inline(always)]
    fn get(&mut self, merges: &MergeLookup, left: u32, right: u32) -> Option<MergeEntry> {
        let key = pair_key(left, right);
        if self.key == Some(key) {
            return self.entry;
        }

        let slot = &mut self.slots[candidate_lookup_cache_slot(key)];
        if slot.valid && slot.key == key {
            self.key = Some(key);
            self.entry = slot.entry;
            return slot.entry;
        }

        let entry = merges.get_key(key);
        self.key = Some(key);
        self.entry = entry;
        *slot = CandidateLookupCacheSlot {
            key,
            valid: true,
            entry,
        };
        entry
    }
}

#[inline(always)]
fn candidate_lookup_cache_slot(key: PairKey) -> usize {
    let mixed = key ^ (key >> 32) ^ (key >> 16);
    (mixed as usize) & (CANDIDATE_LOOKUP_CACHE_SLOTS - 1)
}

#[derive(Default)]
struct OutputState {
    has_previous: bool,
    token: u32,
    old_end: usize,
    merged: bool,
}

fn apply_candidate_merge_level(
    merges: &MergeLookup,
    level: u16,
    min_next_level: usize,
    buffers: &mut TokenizationBuffers,
) -> Option<usize> {
    let token_count = buffers.tokens.len();
    let pair_count = token_count.saturating_sub(1);
    let mut best = NO_CANDIDATE_LEVEL;
    let min_next_level = u16::try_from(min_next_level).unwrap_or(NO_CANDIDATE_LEVEL);

    {
        let tokens = &buffers.tokens;
        let candidate_levels = &buffers.candidate_levels;
        let candidate_new_tokens = &buffers.candidate_new_tokens;
        let next = &mut buffers.next;
        let next_candidate_levels = &mut buffers.next_candidate_levels;
        let next_candidate_new_tokens = &mut buffers.next_candidate_new_tokens;

        next.clear();
        next_candidate_levels.clear();
        next_candidate_new_tokens.clear();
        next.reserve(token_count);
        next_candidate_levels.reserve(pair_count);
        next_candidate_new_tokens.reserve(pair_count);

        let scan_sequentially = should_scan_level_sequentially(candidate_levels, level);

        if scan_sequentially {
            best = apply_candidate_merge_level_dense(
                merges,
                level,
                min_next_level,
                tokens,
                candidate_levels,
                candidate_new_tokens,
                next,
                next_candidate_levels,
                next_candidate_new_tokens,
            );
        } else {
            let mut previous = None;
            let mut lookup_cache = CandidateLookupCache::default();
            let mut index = 0;
            while index < token_count {
                if index >= pair_count {
                    append_copied_range(
                        index,
                        token_count,
                        tokens,
                        candidate_levels,
                        candidate_new_tokens,
                        merges,
                        min_next_level,
                        next,
                        next_candidate_levels,
                        next_candidate_new_tokens,
                        &mut previous,
                        &mut lookup_cache,
                        &mut best,
                    );
                    break;
                }

                let Some(merge_offset) = candidate_levels[index..pair_count]
                    .iter()
                    .position(|candidate| *candidate == level)
                else {
                    append_copied_range(
                        index,
                        token_count,
                        tokens,
                        candidate_levels,
                        candidate_new_tokens,
                        merges,
                        min_next_level,
                        next,
                        next_candidate_levels,
                        next_candidate_new_tokens,
                        &mut previous,
                        &mut lookup_cache,
                        &mut best,
                    );
                    break;
                };

                let merge_index = index + merge_offset;
                append_copied_range(
                    index,
                    merge_index,
                    tokens,
                    candidate_levels,
                    candidate_new_tokens,
                    merges,
                    min_next_level,
                    next,
                    next_candidate_levels,
                    next_candidate_new_tokens,
                    &mut previous,
                    &mut lookup_cache,
                    &mut best,
                );

                append_output_token(
                    OutputToken::merged(merge_index, candidate_new_tokens[merge_index]),
                    candidate_levels,
                    candidate_new_tokens,
                    merges,
                    min_next_level,
                    next,
                    next_candidate_levels,
                    next_candidate_new_tokens,
                    &mut previous,
                    &mut lookup_cache,
                    &mut best,
                );
                index = merge_index + 2;
            }
        }
    }

    std::mem::swap(&mut buffers.tokens, &mut buffers.next);
    std::mem::swap(
        &mut buffers.candidate_levels,
        &mut buffers.next_candidate_levels,
    );
    std::mem::swap(
        &mut buffers.candidate_new_tokens,
        &mut buffers.next_candidate_new_tokens,
    );

    (best != NO_CANDIDATE_LEVEL).then_some(best as usize)
}

#[allow(clippy::too_many_arguments)]
fn apply_candidate_merge_level_dense(
    merges: &MergeLookup,
    level: u16,
    min_next_level: u16,
    tokens: &[u32],
    candidate_levels: &[u16],
    candidate_new_tokens: &[u32],
    next: &mut Vec<u32>,
    next_candidate_levels: &mut Vec<u16>,
    next_candidate_new_tokens: &mut Vec<u32>,
) -> u16 {
    let token_count = tokens.len();
    let pair_count = token_count.saturating_sub(1);
    let mut best = NO_CANDIDATE_LEVEL;
    let mut previous = OutputState::default();
    let mut lookup_cache = CandidateLookupCache::default();
    let mut index = 0;

    while index < token_count {
        if index < pair_count && candidate_levels[index] == level {
            append_merged_output_token(
                candidate_new_tokens[index],
                index,
                merges,
                min_next_level,
                next,
                next_candidate_levels,
                next_candidate_new_tokens,
                &mut previous,
                &mut lookup_cache,
                &mut best,
            );
            index += 2;
        } else {
            append_copied_output_token(
                tokens[index],
                index,
                candidate_levels,
                candidate_new_tokens,
                merges,
                min_next_level,
                next,
                next_candidate_levels,
                next_candidate_new_tokens,
                &mut previous,
                &mut lookup_cache,
                &mut best,
            );
            index += 1;
        }
    }

    best
}

fn should_scan_level_sequentially(candidate_levels: &[u16], level: u16) -> bool {
    const SAMPLE: usize = 1024;
    const MIN_SAMPLE: usize = 64;
    const DENSE_DIVISOR: usize = 16;

    let sample_len = candidate_levels.len().min(SAMPLE);
    if sample_len < MIN_SAMPLE {
        return true;
    }

    let dense_threshold = (sample_len / DENSE_DIVISOR).max(1);
    let mut matches = 0;
    for candidate in candidate_levels.iter().take(sample_len) {
        if *candidate == level {
            matches += 1;
            if matches >= dense_threshold {
                return true;
            }
        }
    }

    false
}

#[allow(clippy::too_many_arguments)]
fn append_copied_range(
    mut start: usize,
    end: usize,
    tokens: &[u32],
    candidate_levels: &[u16],
    candidate_new_tokens: &[u32],
    merges: &MergeLookup,
    min_next_level: u16,
    next: &mut Vec<u32>,
    next_candidate_levels: &mut Vec<u16>,
    next_candidate_new_tokens: &mut Vec<u32>,
    previous: &mut Option<OutputToken>,
    lookup_cache: &mut CandidateLookupCache,
    best: &mut u16,
) {
    if start >= end {
        return;
    }

    append_output_token(
        OutputToken::copied(start, tokens[start]),
        candidate_levels,
        candidate_new_tokens,
        merges,
        min_next_level,
        next,
        next_candidate_levels,
        next_candidate_new_tokens,
        previous,
        lookup_cache,
        best,
    );
    start += 1;

    if start >= end {
        return;
    }

    let candidate_start = start - 1;
    let candidate_end = end - 1;
    append_candidate_range(
        &candidate_levels[candidate_start..candidate_end],
        &candidate_new_tokens[candidate_start..candidate_end],
        min_next_level,
        next_candidate_levels,
        next_candidate_new_tokens,
        best,
    );
    next.extend_from_slice(&tokens[start..end]);
    *previous = Some(OutputToken::copied(end - 1, tokens[end - 1]));
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn append_output_token(
    output: OutputToken,
    candidate_levels: &[u16],
    candidate_new_tokens: &[u32],
    merges: &MergeLookup,
    min_next_level: u16,
    next: &mut Vec<u32>,
    next_candidate_levels: &mut Vec<u16>,
    next_candidate_new_tokens: &mut Vec<u32>,
    previous: &mut Option<OutputToken>,
    lookup_cache: &mut CandidateLookupCache,
    best: &mut u16,
) {
    if let Some(previous) = *previous {
        if !previous.merged
            && !output.merged
            && output.old_start == previous.old_end.saturating_add(1)
        {
            let candidate_index = previous.old_end;
            append_candidate(
                candidate_levels[candidate_index],
                candidate_new_tokens[candidate_index],
                min_next_level,
                next_candidate_levels,
                next_candidate_new_tokens,
                best,
            );
        } else {
            append_lookup_candidate(
                merges,
                previous.token,
                output.token,
                min_next_level,
                next_candidate_levels,
                next_candidate_new_tokens,
                lookup_cache,
                best,
            );
        }
    }

    next.push(output.token);
    *previous = Some(output);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn append_copied_output_token(
    token: u32,
    old_index: usize,
    candidate_levels: &[u16],
    candidate_new_tokens: &[u32],
    merges: &MergeLookup,
    min_next_level: u16,
    next: &mut Vec<u32>,
    next_candidate_levels: &mut Vec<u16>,
    next_candidate_new_tokens: &mut Vec<u32>,
    previous: &mut OutputState,
    lookup_cache: &mut CandidateLookupCache,
    best: &mut u16,
) {
    if previous.has_previous {
        if !previous.merged && old_index == previous.old_end + 1 {
            let candidate_index = previous.old_end;
            append_candidate(
                candidate_levels[candidate_index],
                candidate_new_tokens[candidate_index],
                min_next_level,
                next_candidate_levels,
                next_candidate_new_tokens,
                best,
            );
        } else {
            append_lookup_candidate(
                merges,
                previous.token,
                token,
                min_next_level,
                next_candidate_levels,
                next_candidate_new_tokens,
                lookup_cache,
                best,
            );
        }
    }

    next.push(token);
    previous.has_previous = true;
    previous.token = token;
    previous.old_end = old_index;
    previous.merged = false;
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn append_merged_output_token(
    token: u32,
    old_start: usize,
    merges: &MergeLookup,
    min_next_level: u16,
    next: &mut Vec<u32>,
    next_candidate_levels: &mut Vec<u16>,
    next_candidate_new_tokens: &mut Vec<u32>,
    previous: &mut OutputState,
    lookup_cache: &mut CandidateLookupCache,
    best: &mut u16,
) {
    if previous.has_previous {
        append_lookup_candidate(
            merges,
            previous.token,
            token,
            min_next_level,
            next_candidate_levels,
            next_candidate_new_tokens,
            lookup_cache,
            best,
        );
    }

    next.push(token);
    previous.has_previous = true;
    previous.token = token;
    previous.old_end = old_start + 1;
    previous.merged = true;
}

fn append_candidate_range(
    levels: &[u16],
    new_tokens: &[u32],
    min_next_level: u16,
    next_candidate_levels: &mut Vec<u16>,
    next_candidate_new_tokens: &mut Vec<u32>,
    best: &mut u16,
) {
    if let Some(level) = next_candidate_level_u16(levels, min_next_level) {
        *best = (*best).min(level);
    }

    next_candidate_levels.extend_from_slice(levels);
    next_candidate_new_tokens.extend_from_slice(new_tokens);
}

fn append_lookup_candidate(
    merges: &MergeLookup,
    left: u32,
    right: u32,
    min_next_level: u16,
    next_candidate_levels: &mut Vec<u16>,
    next_candidate_new_tokens: &mut Vec<u32>,
    lookup_cache: &mut CandidateLookupCache,
    best: &mut u16,
) {
    if let Some(entry) = lookup_cache.get(merges, left, right) {
        append_candidate(
            entry.level,
            entry.new_token,
            min_next_level,
            next_candidate_levels,
            next_candidate_new_tokens,
            best,
        );
    } else {
        append_candidate(
            NO_CANDIDATE_LEVEL,
            0,
            min_next_level,
            next_candidate_levels,
            next_candidate_new_tokens,
            best,
        );
    }
}

fn append_candidate(
    level: u16,
    new_token: u32,
    min_next_level: u16,
    next_candidate_levels: &mut Vec<u16>,
    next_candidate_new_tokens: &mut Vec<u32>,
    best: &mut u16,
) {
    if level >= min_next_level && level < *best {
        *best = level;
    }
    next_candidate_levels.push(level);
    next_candidate_new_tokens.push(new_token);
}

fn next_candidate_level_u16(levels: &[u16], min_level: u16) -> Option<u16> {
    let mut best = NO_CANDIDATE_LEVEL;

    for level in levels.iter().copied() {
        if level >= min_level && level < best {
            best = level;
        }
    }

    (best != NO_CANDIDATE_LEVEL).then_some(best)
}

fn best_merge(merges: &FxHashMap<PairKey, MergeRule>, tokens: &[u32]) -> Option<(usize, u32)> {
    if tokens.len() < 2 {
        return None;
    }

    let mut best_rank = u32::MAX;
    let mut best_index = usize::MAX;
    let mut best_token = 0;

    for index in 0..tokens.len() - 1 {
        let pair = pair_key(tokens[index], tokens[index + 1]);
        if let Some(rule) = merges.get(&pair) {
            if rule.rank < best_rank {
                best_rank = rule.rank;
                best_index = index;
                best_token = rule.new_token;
            }
        }
    }

    (best_index != usize::MAX).then_some((best_index, best_token))
}

fn assign_merge_levels(merges: &[RawMerge]) -> Vec<u16> {
    let token_to_merge = merges
        .iter()
        .copied()
        .map(|merge| (merge.new_token, merge))
        .collect::<FxHashMap<_, _>>();

    let mut merge_levels = vec![0; merges.len()];
    let mut left_token_levels = FxHashMap::<u32, u16>::default();
    let mut right_token_levels = FxHashMap::<u32, u16>::default();
    let mut new_token_levels = FxHashMap::<u32, u16>::default();
    let mut tokens_used_to_create_merge = FxHashSet::default();
    let mut stack = Vec::new();

    for (merge_index, merge) in merges.iter().copied().enumerate() {
        let mut level = 0;

        if let Some(conflict_level) = right_token_levels.get(&merge.pair.left).copied() {
            level = level.max(conflict_level.saturating_add(1));
        }
        if let Some(conflict_level) = left_token_levels.get(&merge.pair.right).copied() {
            level = level.max(conflict_level.saturating_add(1));
        }

        tokens_used_to_create_merge.clear();
        stack.clear();
        stack.push(merge.pair.left);
        stack.push(merge.pair.right);

        while let Some(token) = stack.pop() {
            if let Some(child_merge) = token_to_merge.get(&token) {
                if !tokens_used_to_create_merge.insert(token) {
                    continue;
                }
                if let Some(dependency_level) = new_token_levels.get(&token).copied() {
                    level = level.max(dependency_level.saturating_add(1));
                }
                stack.push(child_merge.pair.left);
                stack.push(child_merge.pair.right);
            }
        }

        merge_levels[merge_index] = level;
        left_token_levels
            .entry(merge.pair.left)
            .and_modify(|previous| *previous = (*previous).max(level))
            .or_insert(level);
        right_token_levels
            .entry(merge.pair.right)
            .and_modify(|previous| *previous = (*previous).max(level))
            .or_insert(level);
        new_token_levels.insert(merge.new_token, level);
    }

    merge_levels
}

fn decode_token_bytes(token: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    for ch in token.chars() {
        if let Some(byte) = byte_level_char_to_byte(ch) {
            bytes.push(byte);
        } else {
            let mut buffer = [0; 4];
            bytes.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
        }
    }
    bytes
}

fn byte_level_char_to_byte(ch: char) -> Option<u8> {
    let codepoint = ch as u32;
    if (33..=126).contains(&codepoint)
        || (161..=172).contains(&codepoint)
        || (174..=255).contains(&codepoint)
    {
        return Some(codepoint as u8);
    }

    let mut mapped = 256;
    for byte in 0..=255 {
        if !(33..=126).contains(&byte)
            && !(161..=172).contains(&byte)
            && !(174..=255).contains(&byte)
        {
            if codepoint == mapped {
                return Some(byte as u8);
            }
            mapped += 1;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_tokenizer(ignore_merges: bool) -> FastBpe {
        let vocab = [
            ("a", 0),
            ("b", 1),
            ("c", 2),
            ("d", 3),
            ("ab", 4),
            ("bc", 5),
            ("cd", 6),
            ("abc", 7),
            ("abcd", 8),
        ]
        .into_iter()
        .map(|(token, id)| (token.to_owned(), id));

        FastBpe::from_vocab_and_merges(
            vocab,
            [
                "a b".to_owned(),
                "b c".to_owned(),
                "c d".to_owned(),
                "ab c".to_owned(),
                "abc d".to_owned(),
            ],
            ignore_merges,
        )
        .unwrap()
    }

    #[test]
    fn tokenization_matches_reference() {
        let tokenizer = small_tokenizer(false);
        assert_eq!(
            tokenizer.tokenize(b"abcdabcdbc").unwrap(),
            tokenizer.tokenize_reference(b"abcdabcdbc").unwrap()
        );
    }

    #[test]
    fn competing_merges_are_resolved_by_rank() {
        let tokenizer = small_tokenizer(false);
        assert_eq!(tokenizer.tokenize(b"abc").unwrap(), vec![7]);
    }

    #[test]
    fn ignore_merges_uses_direct_vocab_match() {
        let tokenizer = small_tokenizer(true);
        assert_eq!(tokenizer.tokenize(b"abcd").unwrap(), vec![8]);
    }

    #[test]
    fn missing_byte_is_reported() {
        let tokenizer = small_tokenizer(false);
        let err = tokenizer.tokenize(b"z").unwrap_err();
        assert!(matches!(err, TokenizerError::MissingByteToken(b'z')));
    }

    #[test]
    fn byte_level_token_glyphs_decode_to_original_bytes() {
        assert_eq!(decode_token_bytes("Ġ"), b" ");
        assert_eq!(decode_token_bytes("Ċ"), b"\n");
        assert_eq!(decode_token_bytes("Ā"), &[0]);
    }

    #[test]
    fn levelized_tokenization_matches_reference_for_long_input() {
        let tokenizer = small_tokenizer(false);
        let input = "abcd".repeat(80).into_bytes();
        assert_eq!(
            tokenizer.tokenize(&input).unwrap(),
            tokenizer.tokenize_reference(&input).unwrap()
        );
    }

    #[test]
    fn levelized_self_merge_matches_reference_for_long_input() {
        let vocab = [("a", 0), ("b", 1), ("c", 2), ("aa", 3), ("bc", 4)]
            .into_iter()
            .map(|(token, id)| (token.to_string(), id));
        let merges = ["a a", "b c"].into_iter().map(str::to_string);
        let tokenizer = FastBpe::from_vocab_and_merges(vocab, merges, false).unwrap();
        let input = vec![b'a'; 16 * 1024 + 257];

        assert_eq!(
            tokenizer.tokenize(&input).unwrap(),
            tokenizer.tokenize_reference(&input).unwrap()
        );
    }

    #[test]
    fn merge_levels_do_not_have_adjacent_pair_conflicts() {
        let tokenizer = random_tokenizer(&mut Lcg::new(0xdead_beef));

        for level in 0..tokenizer.levels.len() as u16 {
            let pairs = tokenizer
                .level_merges
                .entries
                .iter()
                .filter(|entry| entry.level == level)
                .map(|entry| split_pair_key(entry.key))
                .collect::<Vec<_>>();
            for (index, left) in pairs.iter().copied().enumerate() {
                for right in pairs.iter().copied().skip(index + 1) {
                    assert_ne!(left.1, right.0, "{left:?} can overlap {right:?}");
                    assert_ne!(right.1, left.0, "{right:?} can overlap {left:?}");
                }
            }
        }
    }

    #[test]
    fn batch_tokenization_matches_sequential_concatenation() {
        let tokenizer = small_tokenizer(false);
        let inputs: [&[u8]; 5] = [b"abcd", b"abc", b"", b"bc", b"abcdabcdbc"];

        let mut buffers = BatchTokenizationBuffers::default();
        let batch = tokenizer
            .tokenize_batch_into(&inputs, &mut buffers)
            .unwrap()
            .to_vec();

        assert_eq!(batch, sequential_batch(&tokenizer, &inputs));
        assert_eq!(buffers.tokens(), batch);
    }

    #[test]
    fn batch_buffers_can_be_reused() {
        let tokenizer = small_tokenizer(false);
        let mut buffers = BatchTokenizationBuffers::default();

        let first_inputs: [&[u8]; 2] = [b"abcd", b"bc"];
        let first = tokenizer
            .tokenize_batch_into(&first_inputs, &mut buffers)
            .unwrap()
            .to_vec();
        assert_eq!(first, sequential_batch(&tokenizer, &first_inputs));

        let second_inputs: [&[u8]; 3] = [b"abc", b"", b"abcdabcdbc"];
        let second = tokenizer
            .tokenize_batch_into(&second_inputs, &mut buffers)
            .unwrap()
            .to_vec();
        assert_eq!(second, sequential_batch(&tokenizer, &second_inputs));
    }

    #[test]
    fn randomized_levelized_tokenization_matches_greedy_reference() {
        let mut rng = Lcg::new(0x1234_5678);

        for _ in 0..100 {
            let tokenizer = random_tokenizer(&mut rng);
            let mut buffers = TokenizationBuffers::default();

            for _ in 0..100 {
                let len = 1 + rng.usize(192);
                let input = (0..len)
                    .map(|_| b'a' + rng.usize(4) as u8)
                    .collect::<Vec<_>>();

                let fast = tokenizer
                    .tokenize_into(&input, &mut buffers)
                    .unwrap()
                    .to_vec();
                let reference = tokenizer.tokenize_reference(&input).unwrap();
                assert_eq!(
                    fast,
                    reference,
                    "input: {:?}",
                    String::from_utf8_lossy(&input)
                );
            }
        }
    }

    fn sequential_batch(tokenizer: &FastBpe, inputs: &[&[u8]]) -> Vec<u32> {
        let mut buffers = TokenizationBuffers::default();
        let mut out = Vec::new();
        for input in inputs {
            out.extend_from_slice(tokenizer.tokenize_into(input, &mut buffers).unwrap());
        }
        out
    }

    fn random_tokenizer(rng: &mut Lcg) -> FastBpe {
        let mut token_strings = ["a", "b", "c", "d"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut vocab = token_strings
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect::<FxHashMap<_, _>>();
        let mut merges = Vec::new();

        while merges.len() < 40 {
            let left = token_strings[rng.usize(token_strings.len())].clone();
            let right = token_strings[rng.usize(token_strings.len())].clone();
            let merged = format!("{left}{right}");
            if merged.len() > 24 || vocab.contains_key(&merged) {
                continue;
            }

            let id = vocab.len() as u32;
            vocab.insert(merged.clone(), id);
            token_strings.push(merged);
            merges.push(format!("{left} {right}"));
        }

        FastBpe::from_vocab_and_merges(vocab, merges, false).unwrap()
    }

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            self.state
        }

        fn usize(&mut self, upper_bound: usize) -> usize {
            (self.next() as usize) % upper_bound
        }
    }
}
