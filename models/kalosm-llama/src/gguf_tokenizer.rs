use std::{cell::RefCell, collections::HashMap};

use kalosm_tokenizer::{FastBpe, TokenizationBuffers};
use regex::Regex;
use thiserror::Error;

thread_local! {
    static ENCODE_BUFFERS: RefCell<GgufEncodeBuffers> = RefCell::new(GgufEncodeBuffers::default());
}

#[derive(Clone, Copy)]
enum PreTokenizerType {
    Bloom,
    Chameleon,
    Chatglm4,
    Codeshell,
    CommandR,
    Dbrx,
    DeepseekCoder,
    DeepseekLlm,
    Default,
    Exaone,
    Falcon,
    Gpt2,
    Gpt3Finnish,
    Jais,
    Llama3,
    Minerva,
    Mpt,
    Olmo,
    Poro,
    Qwen2,
    Refact,
    Smaug,
    Smollm,
    Stablelm2,
    Starcoder,
    Tekken,
    Viking,
}

impl PreTokenizerType {
    // Adapted from llama.cpp and kept in sync with the previous tokenizers-backed builder.
    const fn regexes(self) -> &'static [&'static str] {
        match self {
            Self::Llama3 => &[
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            ],
            Self::Dbrx | Self::Smaug => &[
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            ],
            Self::DeepseekLlm => &[
                "[\r\n]",
                "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                "\\s+$",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}+",
            ],
            Self::DeepseekCoder => &[
                "[\r\n]",
                "\\s?\\p{L}+",
                "\\s?\\p{P}+",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}",
            ],
            Self::Falcon => &[
                "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                "[0-9][0-9][0-9]",
            ],
            Self::Starcoder
            | Self::Refact
            | Self::CommandR
            | Self::Smollm
            | Self::Codeshell
            | Self::Exaone
            | Self::Minerva => &[
                "\\p{N}",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            ],
            Self::Gpt2 | Self::Mpt | Self::Olmo | Self::Jais => &[
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            ],
            Self::Stablelm2 | Self::Qwen2 => &[
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            ],
            Self::Poro | Self::Bloom | Self::Gpt3Finnish => {
                &[" ?[^(\\s|.,!?…。，、।۔،)]+"]
            }
            Self::Chatglm4 => &[
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            ],
            Self::Viking => &[" ?[^(\\s|.,!?…。，、।۔،)]+", "\\p{N}"],
            Self::Tekken => &[
                "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            ],
            Self::Chameleon => &[
                "<sentinel:[0-9]+>",
                "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
                "([\\t\\n]|    |  )",
                "\\p{N}",
                "[\\p{P}!-/:-@\\[-`{-~]",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            ],
            Self::Default => &[
                "[\\p{P}\\$\\+<=>\\^~\\|]+",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                "\\p{N}+",
                "[0-9][0-9][0-9]",
            ],
        }
    }
}

#[derive(Clone)]
struct PreTokenizer {
    regexes: Vec<SplitRegex>,
}

#[derive(Clone, Copy)]
struct TextRange {
    start: usize,
    end: usize,
}

impl TextRange {
    #[inline(always)]
    fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    #[inline(always)]
    fn is_empty(self) -> bool {
        self.start == self.end
    }

    #[inline(always)]
    fn as_str(self, text: &str) -> &str {
        &text[self.start..self.end]
    }
}

#[derive(Default)]
struct PreTokenizationBuffers {
    pieces: Vec<TextRange>,
    next: Vec<TextRange>,
}

#[derive(Default)]
struct GgufEncodeBuffers {
    pre_tokenizer: PreTokenizationBuffers,
    tokenization: TokenizationBuffers,
}

#[derive(Clone)]
struct SplitRegex {
    regex: Regex,
    trim_trailing_horizontal_space_before_nonspace: bool,
    preserve_newline_runs: bool,
}

impl PreTokenizer {
    fn new(ty: PreTokenizerType) -> Result<Self, GgufTokenizerError> {
        let regexes = ty
            .regexes()
            .iter()
            .map(|regex| {
                Ok(SplitRegex {
                    trim_trailing_horizontal_space_before_nonspace: regex.contains("\\s+(?!\\S)"),
                    preserve_newline_runs: regex.contains("\\r\\n")
                        || regex.contains("[\r\n]")
                        || regex.contains("\\n"),
                    regex: Regex::new(&sanitize_regex(regex))?,
                })
            })
            .collect::<Result<Vec<_>, regex::Error>>()?;
        Ok(Self { regexes })
    }

    fn split_into_ranges(&self, text: &str, buffers: &mut PreTokenizationBuffers) {
        buffers.pieces.clear();
        buffers.next.clear();
        buffers.pieces.push(TextRange::new(0, text.len()));

        for split_regex in &self.regexes {
            buffers.next.clear();
            for piece in buffers.pieces.drain(..) {
                split_piece(split_regex, text, piece, &mut buffers.next);
            }
            std::mem::swap(&mut buffers.pieces, &mut buffers.next);
        }
    }
}

fn split_piece(split_regex: &SplitRegex, text: &str, range: TextRange, out: &mut Vec<TextRange>) {
    let text = range.as_str(text);
    let mut offset = 0;
    while offset < text.len() {
        let Some(found) = split_regex.regex.find_at(text, offset) else {
            break;
        };
        if found.start() > offset {
            out.push(TextRange::new(
                range.start + offset,
                range.start + found.start(),
            ));
        }

        let end = adjusted_match_end(split_regex, text, found.start(), found.end());
        if found.start() != end {
            out.push(TextRange::new(
                range.start + found.start(),
                range.start + end,
            ));
        }
        offset = end;
    }
    if offset < text.len() {
        out.push(TextRange::new(range.start + offset, range.end));
    }
}

fn adjusted_match_end(
    split_regex: &SplitRegex,
    text: &str,
    match_start: usize,
    match_end: usize,
) -> usize {
    if !split_regex.trim_trailing_horizontal_space_before_nonspace {
        return match_end;
    }
    let matched = &text[match_start..match_end];
    if matched.is_empty() || !matched.chars().all(char::is_whitespace) {
        return match_end;
    }
    if split_regex.preserve_newline_runs && (matched.contains('\n') || matched.contains('\r')) {
        return match_end;
    }
    let Some(next) = text[match_end..].chars().next() else {
        return match_end;
    };
    if next.is_whitespace() {
        return match_end;
    }
    if matched.chars().nth(1).is_none() {
        return match_end;
    }
    if let Some((last_start, _)) = matched.char_indices().last() {
        match_start + last_start
    } else {
        match_end
    }
}

fn sanitize_regex(regex: &str) -> String {
    regex.replace("\\s+(?!\\S)", "\\s+")
}

// Adapted from tokenizer code in llama.cpp.
#[derive(Clone, Copy)]
pub(crate) struct GGUFPreTokenizerConfig {
    add_bos: bool,
    ignore_merges: bool,
    ty: PreTokenizerType,
}

impl GGUFPreTokenizerConfig {
    pub(crate) fn build(
        &self,
        vocab: HashMap<String, u32>,
        types: Vec<u8>,
        merges: Vec<(String, String)>,
        bos: &str,
        eos: &str,
    ) -> Result<GgufTokenizer, GgufTokenizerError> {
        let bos_token = vocab[bos];
        let eos_token = vocab[eos];
        let max_token = vocab.values().copied().max().unwrap_or(0) as usize;
        let mut special_tokens = vec![false; max_token + 1];
        let mut special_token_matches = vec![Vec::new(); 256];

        for (token, id) in &vocab {
            let token_type = types.get(*id as usize).copied().unwrap_or(1);
            if token_type != 1 || *id == bos_token || *id == eos_token {
                if let Some(special) = special_tokens.get_mut(*id as usize) {
                    *special = true;
                }
                if let Some((&first, _)) = token.as_bytes().split_first() {
                    special_token_matches[first as usize].push((token.as_bytes().to_vec(), *id));
                }
            }
        }
        for bucket in &mut special_token_matches {
            bucket.sort_unstable_by(|(left, _), (right, _)| right.len().cmp(&left.len()));
        }

        let merges = merges
            .into_iter()
            .map(|(left, right)| format!("{left} {right}"));
        let bpe = FastBpe::from_vocab_and_merges(vocab, merges, self.ignore_merges)?;
        let pre_tokenizer = PreTokenizer::new(self.ty)?;

        Ok(GgufTokenizer {
            bpe,
            pre_tokenizer,
            special_tokens,
            special_token_matches,
            bos_token,
            add_bos: self.add_bos,
        })
    }
}

impl Default for GGUFPreTokenizerConfig {
    fn default() -> Self {
        Self {
            add_bos: true,
            ignore_merges: false,
            ty: PreTokenizerType::Default,
        }
    }
}

#[derive(Clone)]
pub(crate) struct GgufTokenizer {
    bpe: FastBpe,
    pre_tokenizer: PreTokenizer,
    special_tokens: Vec<bool>,
    special_token_matches: Vec<Vec<(Vec<u8>, u32)>>,
    bos_token: u32,
    add_bos: bool,
}

impl GgufTokenizer {
    pub(crate) fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, GgufTokenizerError> {
        ENCODE_BUFFERS.with(|buffers| {
            if let Ok(mut buffers) = buffers.try_borrow_mut() {
                self.encode_with_buffers(text, add_special_tokens, &mut buffers)
            } else {
                let mut buffers = GgufEncodeBuffers::default();
                self.encode_with_buffers(text, add_special_tokens, &mut buffers)
            }
        })
    }

    fn encode_with_buffers(
        &self,
        text: &str,
        add_special_tokens: bool,
        buffers: &mut GgufEncodeBuffers,
    ) -> Result<Vec<u32>, GgufTokenizerError> {
        let mut out = Vec::with_capacity(
            (text.len() / 4).max(8) + usize::from(add_special_tokens && self.add_bos),
        );
        if add_special_tokens && self.add_bos {
            out.push(self.bos_token);
        }

        let bytes = text.as_bytes();
        let mut offset = 0;
        let mut normal_start = 0;
        while offset < bytes.len() {
            if let Some((len, token)) = self.match_special(bytes, offset) {
                self.encode_normal(&text[normal_start..offset], &mut out, buffers)?;
                out.push(token);
                offset += len;
                normal_start = offset;
            } else {
                let ch = text[offset..].chars().next().expect("offset is in bounds");
                offset += ch.len_utf8();
            }
        }
        self.encode_normal(&text[normal_start..], &mut out, buffers)?;

        Ok(out)
    }

    pub(crate) fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> String {
        let mut bytes = Vec::new();
        for token in tokens {
            if skip_special_tokens && self.is_special_token(*token) {
                continue;
            }
            if let Some(token_bytes) = self.bpe.token_bytes(*token) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub(crate) fn is_special_token(&self, token: u32) -> bool {
        self.special_tokens
            .get(token as usize)
            .copied()
            .unwrap_or(false)
    }

    fn encode_normal(
        &self,
        text: &str,
        out: &mut Vec<u32>,
        buffers: &mut GgufEncodeBuffers,
    ) -> Result<(), GgufTokenizerError> {
        if text.is_empty() {
            return Ok(());
        }

        let GgufEncodeBuffers {
            pre_tokenizer,
            tokenization,
        } = buffers;

        self.pre_tokenizer.split_into_ranges(text, pre_tokenizer);
        for piece in pre_tokenizer.pieces.iter().copied() {
            if piece.is_empty() {
                continue;
            }

            let tokenized = self
                .bpe
                .tokenize_into(piece.as_str(text).as_bytes(), tokenization)?;
            out.extend_from_slice(tokenized);
        }

        Ok(())
    }

    fn match_special(&self, bytes: &[u8], offset: usize) -> Option<(usize, u32)> {
        let remaining = &bytes[offset..];
        let first = *remaining.first()?;
        self.special_token_matches[first as usize]
            .iter()
            .find_map(|(token, id)| remaining.starts_with(token).then_some((token.len(), *id)))
    }
}

#[derive(Debug, Error)]
pub(crate) enum GgufTokenizerError {
    #[error("byte-level BPE error: {0}")]
    Bpe(#[from] kalosm_tokenizer::TokenizerError),
    #[error("pre-tokenizer regex error: {0}")]
    Regex(#[from] regex::Error),
}

pub(crate) fn get_pre_tokenizer(
    pre_tokenizer_type: &str,
    add_bos: Option<bool>,
) -> GGUFPreTokenizerConfig {
    let mut tokenizer = match pre_tokenizer_type {
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" => GGUFPreTokenizerConfig {
            ignore_merges: true,
            add_bos: true,
            ty: PreTokenizerType::Llama3,
        },
        "deepseek-llm" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::DeepseekLlm,
            ..Default::default()
        },
        "deepseek-coder" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::DeepseekCoder,
            ..Default::default()
        },
        "falcon" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Falcon,
            ..Default::default()
        },
        "mpt" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Mpt,
            ..Default::default()
        },
        "starcoder" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Starcoder,
            ..Default::default()
        },
        "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v1-en" | "jina-v2-es"
        | "jina-v2-de" | "jina-v2-code" | "roberta-bpe" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Gpt2,
            ..Default::default()
        },
        "refact" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Refact,
            ..Default::default()
        },
        "command-r" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::CommandR,
            ..Default::default()
        },
        "qwen2" | "qwen3" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Qwen2,
            ..Default::default()
        },
        "stablelm2" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Stablelm2,
            ..Default::default()
        },
        "olmo" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Olmo,
            ..Default::default()
        },
        "dbrx" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Dbrx,
            ..Default::default()
        },
        "smaug-bpe" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Smaug,
            ..Default::default()
        },
        "poro-chat" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Poro,
            ..Default::default()
        },
        "chatglm-bpe" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Chatglm4,
            ..Default::default()
        },
        "viking" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Viking,
            ..Default::default()
        },
        "jais" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Jais,
            ..Default::default()
        },
        "tekken" => GGUFPreTokenizerConfig {
            ignore_merges: true,
            add_bos: true,
            ty: PreTokenizerType::Tekken,
        },
        "smollm" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Smollm,
            ..Default::default()
        },
        "codeshell" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Codeshell,
            ..Default::default()
        },
        "bloom" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Bloom,
            ..Default::default()
        },
        "gpt3-finnish" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Gpt3Finnish,
            ..Default::default()
        },
        "exaone" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Exaone,
            ..Default::default()
        },
        "chameleon" => GGUFPreTokenizerConfig {
            add_bos: true,
            ty: PreTokenizerType::Chameleon,
            ..Default::default()
        },
        "minerva-7b" => GGUFPreTokenizerConfig {
            ty: PreTokenizerType::Minerva,
            ..Default::default()
        },

        _ => GGUFPreTokenizerConfig::default(),
    };

    if let Some(add_bos) = add_bos {
        tokenizer.add_bos = add_bos;
    }

    tokenizer
}

#[cfg(all(test, feature = "hf-tokenizer-json"))]
mod tests {
    use super::*;
    use tokenizers::decoders::byte_level::ByteLevel;
    use tokenizers::models::bpe::BpeBuilder;
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::processors::template::{SpecialToken, TemplateProcessing};
    use tokenizers::AddedToken;
    use tokenizers::{
        OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer as TokenizersPreTokenizer,
        SplitDelimiterBehavior,
    };

    const LABELS: &[&str] = &[
        "llama3",
        "llama-v3",
        "llama-bpe",
        "falcon3",
        "deepseek-llm",
        "deepseek-coder",
        "falcon",
        "mpt",
        "starcoder",
        "gpt-2",
        "phi-2",
        "jina-es",
        "jina-de",
        "gigachat",
        "jina-v1-en",
        "jina-v2-es",
        "jina-v2-de",
        "jina-v2-code",
        "roberta-bpe",
        "refact",
        "command-r",
        "qwen2",
        "qwen3",
        "stablelm2",
        "olmo",
        "dbrx",
        "smaug-bpe",
        "poro-chat",
        "chatglm-bpe",
        "viking",
        "jais",
        "tekken",
        "smollm",
        "codeshell",
        "bloom",
        "gpt3-finnish",
        "exaone",
        "chameleon",
        "minerva-7b",
        "unknown-default",
    ];

    const SAMPLES: &[&str] = &[
        "Hello, world! 123",
        " can't we'll I'M you're they're",
        "foo   bar\n\nbaz\tqux   ",
        "中文かな 한글 русский Ελληνικά",
        "IMGIMGABCDZ <sentinel:42>    \t\n",
        "code::{ let x = 12345; }\r\nnext",
    ];

    #[test]
    fn native_pretokenizers_match_tokenizers_split_sequence() {
        for label in LABELS {
            let config = get_pre_tokenizer(label, None);
            let native = super::PreTokenizer::new(config.ty).unwrap();
            let legacy = legacy_pre_tokenizer(config.ty).unwrap();

            for sample in SAMPLES {
                let native_splits = native.split(sample);
                let legacy_splits = legacy_splits(&legacy, sample).unwrap();
                assert_eq!(
                    native_splits, legacy_splits,
                    "pre-tokenizer label `{label}` mismatched for sample `{sample}`"
                );
            }
        }
    }

    #[test]
    fn native_gguf_encoding_matches_tokenizers_byte_level_fixture() {
        for label in LABELS {
            let config = get_pre_tokenizer(label, None);
            let (vocab, types, bos, eos) = byte_vocab();
            let native = config
                .build(vocab.clone(), types.clone(), Vec::new(), &bos, &eos)
                .unwrap();
            let legacy = legacy_tokenizer(&config, vocab, types, Vec::new(), &bos, &eos).unwrap();

            for sample in SAMPLES
                .iter()
                .copied()
                .chain(["<s>Hello</s>   tail", "tabs\t\tword and spaces   word"])
            {
                for add_special_tokens in [false, true] {
                    let native_ids = native.encode(sample, add_special_tokens).unwrap();
                    let legacy_ids = legacy
                        .encode_fast(sample, add_special_tokens)
                        .unwrap()
                        .get_ids()
                        .to_vec();
                    assert_eq!(
                        native_ids, legacy_ids,
                        "encoded ids mismatch for label `{label}`, add_special_tokens={add_special_tokens}, sample `{sample}`"
                    );

                    for skip_special_tokens in [false, true] {
                        let native_text = native.decode(&native_ids, skip_special_tokens);
                        let legacy_text = legacy.decode(&legacy_ids, skip_special_tokens).unwrap();
                        assert_eq!(
                            native_text, legacy_text,
                            "decoded text mismatch for label `{label}`, add_special_tokens={add_special_tokens}, skip_special_tokens={skip_special_tokens}, sample `{sample}`"
                        );
                    }
                }
            }
        }
    }

    fn legacy_pre_tokenizer(ty: PreTokenizerType) -> tokenizers::Result<Sequence> {
        let splits = ty
            .regexes()
            .iter()
            .map(|regex| {
                Split::new(
                    SplitPattern::Regex((*regex).to_string()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .map(Into::into)
            })
            .collect::<tokenizers::Result<Vec<_>>>()?;
        Ok(Sequence::new(splits))
    }

    fn legacy_tokenizer(
        config: &GGUFPreTokenizerConfig,
        vocab: HashMap<String, u32>,
        types: Vec<u8>,
        merges: Vec<(String, String)>,
        bos: &str,
        eos: &str,
    ) -> tokenizers::Result<tokenizers::Tokenizer> {
        let mut special_tokens: Vec<_> = vocab
            .iter()
            .filter_map(|(token, id)| {
                if types[*id as usize] == 1 {
                    None
                } else {
                    Some(AddedToken::from(token.to_string(), true))
                }
            })
            .collect();
        let bos_token = vocab[bos];
        let bpe_tokenizer = BpeBuilder::new()
            .vocab_and_merges(ahash::AHashMap::from_iter(vocab.clone()), merges)
            .ignore_merges(config.ignore_merges)
            .build()?;

        let byte_level_pre = ByteLevel::new(false, true, false);
        let byte_level_post = ByteLevel::new(true, false, true);
        let byte_level_decoder = ByteLevel::new(true, true, true);

        let mut tokenizer = tokenizers::Tokenizer::new(bpe_tokenizer);
        let mut pre_tokenizers = Vec::new();
        for regex in config.ty.regexes() {
            let split = Split::new(
                SplitPattern::Regex(regex.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )?;
            pre_tokenizers.push(split.into());
        }
        pre_tokenizers.push(byte_level_pre.into());
        tokenizer.with_pre_tokenizer(Some(Sequence::new(pre_tokenizers)));
        tokenizer.with_decoder(Some(byte_level_decoder));

        let mut post_processors = Vec::new();
        post_processors.push(byte_level_post.into());
        if config.add_bos {
            let special_toks = vec![SpecialToken::from((bos_token, bos.to_string()))];
            post_processors.push(
                TemplateProcessing::builder()
                    .single(tokenizers::processors::template::Template::try_from(vec![
                        format!("{bos}:0"),
                        "$A:0".to_string(),
                    ])?)
                    .pair(tokenizers::processors::template::Template::try_from(vec![
                        format!("{bos}:0"),
                        "$A:0".to_string(),
                        format!("{bos}:1"),
                        "$B:1".to_string(),
                    ])?)
                    .special_tokens(special_toks)
                    .build()?
                    .into(),
            );
        }
        tokenizer.with_post_processor(Some(tokenizers::processors::sequence::Sequence::new(
            post_processors,
        )));
        special_tokens.push(AddedToken::from(bos.to_string(), true));
        special_tokens.push(AddedToken::from(eos.to_string(), true));
        tokenizer.add_special_tokens(&special_tokens);

        Ok(tokenizer)
    }

    fn byte_vocab() -> (HashMap<String, u32>, Vec<u8>, String, String) {
        let mut vocab = HashMap::new();
        let mut types = vec![1; 258];
        for byte in 0..=255 {
            vocab.insert(byte_level_byte_to_char(byte).to_string(), byte as u32);
        }

        let bos = "<s>".to_string();
        let eos = "</s>".to_string();
        vocab.insert(bos.clone(), 256);
        vocab.insert(eos.clone(), 257);
        types[256] = 3;
        types[257] = 3;

        (vocab, types, bos, eos)
    }

    fn byte_level_byte_to_char(byte: u8) -> char {
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte)
        {
            return char::from_u32(byte as u32).unwrap();
        }

        let mut mapped = 256;
        for candidate in 0..=255 {
            if !(33..=126).contains(&candidate)
                && !(161..=172).contains(&candidate)
                && !(174..=255).contains(&candidate)
            {
                if candidate == byte {
                    return char::from_u32(mapped).unwrap();
                }
                mapped += 1;
            }
        }

        unreachable!("all bytes are covered by the byte-level alphabet")
    }

    fn legacy_splits<'a>(
        pre_tokenizer: &Sequence,
        text: &'a str,
    ) -> tokenizers::Result<Vec<String>> {
        let mut pretokenized = PreTokenizedString::from(text);
        pre_tokenizer.pre_tokenize(&mut pretokenized)?;
        Ok(pretokenized
            .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
            .into_iter()
            .map(|(split, _, _)| split.to_string())
            .collect())
    }
}
