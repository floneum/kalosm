use srx::SRX;
use std::cell::OnceCell;
use std::rc::Rc;
use std::str::FromStr;

/// The default sentence chunker. Unlike [`SentenceChunker`], this is Send + Sync.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultSentenceChunker;

impl DefaultSentenceChunker {
    /// Split a string into sentence byte ranges.
    pub fn split_sentences(&self, string: &str) -> Vec<std::ops::Range<usize>> {
        SentenceChunker::default().split_sentences(string)
    }
}

/// A sentence splitter backed by [SRX](https://www.unicode.org/uli/pas/srx/srx20.html) rules.
///
/// Uses the [srx](https://crates.io/crates/srx) crate to parse and apply the rules.
#[derive(Debug, Clone)]
pub struct SentenceChunker {
    srx: Rc<SRX>,
}

impl SentenceChunker {
    /// Create a new sentence chunker from an xml rules string.
    pub fn new(rules: &str) -> Self {
        Self {
            srx: SRX::from_str(rules)
                .expect("the rules file is valid")
                .into(),
        }
    }

    /// Create a new sentence chunker from anything that implements [`std::io::Read`] in the srx rules format.
    pub fn load(reader: impl std::io::Read) -> Result<Self, srx::Error> {
        Ok(Self {
            srx: SRX::from_reader(reader)?.into(),
        })
    }

    /// Split the body of a document into a list of sentence byte ranges.
    pub fn split_sentences(&self, string: &str) -> Vec<std::ops::Range<usize>> {
        let language = whatlang::detect_lang(string)
            .map(|lang_code| lang_code.code())
            .unwrap_or("en");

        let rules = self.srx.language_rules(language);
        rules.split_ranges(string)
    }

    /// Access the parsed SRX rules. Primarily useful to external `Chunker` trait implementations.
    pub fn srx(&self) -> &SRX {
        &self.srx
    }
}

impl Default for SentenceChunker {
    fn default() -> Self {
        // The rules are expensive to parse (~1 second), so cache them in a thread-local.
        thread_local! {
            static DEFAULT_RULES: OnceCell<Rc<SRX>> = const { OnceCell::new() };
        }

        let rules = DEFAULT_RULES.with(|default| {
            default
                .get_or_init(|| {
                    // LanguageTool ruleset: https://github.com/languagetool-org/languagetool/blob/master/languagetool-core/src/main/resources/org/languagetool/resource/segment.srx
                    let rules = SRX::from_str(include_str!("./assets/segment.srx"))
                        .expect("the rules file is valid");
                    Rc::new(rules)
                })
                .clone()
        });

        Self { srx: rules }
    }
}
