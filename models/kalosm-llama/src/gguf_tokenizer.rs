use std::collections::HashMap;

use tokenizers::{
    decoders::byte_level::ByteLevel,
    models::bpe::BpeBuilder,
    pre_tokenizers::split::SplitPattern,
    processors::template::{SpecialToken, TemplateProcessing},
    AddedToken, Tokenizer,
};

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
    // Adapted from llama.cpp https://github.com/ggerganov/llama.cpp/blob/f922a9c542ee117550a168395c63ea79261f5c99/src/llama-vocab.cpp#L355-L481
    const fn regexes(self) -> &'static [&'static str] {
        match self {
            Self::Llama3 => {
                &[
                    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                ]
            },
            Self::Dbrx| Self::Smaug => {
                &["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"]
            },
                     Self::DeepseekLlm => {
                        &[
                            "[\r\n]",
                            "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                            "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                            "\\s+$",
                            "[一-龥ࠀ-一가-퟿]+",
                            "\\p{N}+",
                        ]
                     }
                     Self::DeepseekCoder => {
        &[
                 "[\r\n]",
                 "\\s?\\p{L}+",
                 "\\s?\\p{P}+",
                 "[一-龥ࠀ-一가-퟿]+",
                 "\\p{N}",]}
                     Self::Falcon => {
        &[
                 "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                 "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                 "[0-9][0-9][0-9]",]}
        Self::Starcoder | Self::Refact | Self::CommandR | Self::Smollm | Self::Codeshell | Self::Exaone | Self::Minerva => {
            &[
                "\\p{N}",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            ]
        },
        Self::Gpt2 | Self::Mpt | Self::Olmo | Self::Jais => {
            &[
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            ]
        }
        Self::Stablelm2 | Self::Qwen2 => {
            &[
                "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            ]
        },
        Self::Poro | Self::Bloom | Self::Gpt3Finnish => {
            &[
                " ?[^(\\s|.,!?…。，、।۔،)]+",
            ]
        },
        Self::Chatglm4 => {
            &[
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            ]
        },
        Self::Viking => {
            &[
                " ?[^(\\s|.,!?…。，、।۔،)]+",
                "\\p{N}",
            ]
        },
Self::Tekken => {
            &["[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"]
            }
        Self::Chameleon => {
            &[
                "<sentinel:[0-9]+>",  // Sentinel tokens
                "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",  // Image tokens
                "([\\t\\n]|    |  )",  // directly from tokenizer.json
                "\\p{N}", // Individual digits
                "[\\p{P}!-/:-@\\[-`{-~]",  // Punctuation, Isolated
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            ]
        }
        Self::Default => &[
            "[\\p{P}\\$\\+<=>\\^~\\|]+",
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            "\\p{N}+",
            "[0-9][0-9][0-9]",
        ],}
    }
}

// Adapted from tokenizer code in llama.cpp https://github.com/ggerganov/llama.cpp/blob/f922a9c542ee117550a168395c63ea79261f5c99/src/llama-model.cpp#L1096
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
    ) -> Result<Tokenizer, tokenizers::Error> {
        let mut special_tokens: Vec<_> = vocab
            .iter()
            .filter_map(|(k, v)| {
                // 1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte
                let special_token = types[*v as usize];
                if special_token == 1 {
                    return None;
                }
                Some(AddedToken::from(k.to_string(), true))
            })
            .collect();
        let bos_token = vocab[bos];
        let vocab = ahash::AHashMap::from_iter(vocab);
        let bpe_tokenizer = BpeBuilder::new()
            .vocab_and_merges(vocab, merges)
            .ignore_merges(self.ignore_merges)
            .build()?;

        let byte_level_pre = ByteLevel::new(false, true, false);
        let byte_level_post = ByteLevel::new(true, false, true);
        let byte_level_decoder = ByteLevel::new(true, true, true);

        let mut tokenizer = Tokenizer::new(bpe_tokenizer);
        let mut pre_tokenizers = Vec::new();
        for regex in self.ty.regexes() {
            let split = tokenizers::pre_tokenizers::split::Split::new(
                SplitPattern::Regex(regex.to_string()),
                tokenizers::SplitDelimiterBehavior::Isolated,
                false,
            )?;
            pre_tokenizers.push(split.into());
        }
        pre_tokenizers.push(byte_level_pre.into());
        let pre_tokenizer = tokenizers::pre_tokenizers::sequence::Sequence::new(pre_tokenizers);
        tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
        tokenizer.with_decoder(Some(byte_level_decoder));
        let mut post_processors = Vec::new();
        post_processors.push(byte_level_post.into());
        if self.add_bos {
            let special_toks = vec![SpecialToken::from((bos_token, bos.to_string()))];
            post_processors.push(
                TemplateProcessing::builder()
                    .single(
                        tokenizers::processors::template::Template::try_from(vec![
                            format!("{bos}:0"),
                            "$A:0".to_string(),
                        ])
                        .unwrap(),
                    )
                    .pair(
                        tokenizers::processors::template::Template::try_from(vec![
                            format!("{bos}:0"),
                            "$A:0".to_string(),
                            format!("{bos}:1"),
                            "$B:1".to_string(),
                        ])
                        .unwrap(),
                    )
                    .special_tokens(special_toks)
                    .build()
                    .unwrap()
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
        "qwen2" => GGUFPreTokenizerConfig {
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
