//! Text chunking primitives used across kalosm crates.
//!
//! This crate provides [`ChunkStrategy`] (paragraph / sentence / word sliding windows) and
//! [`SentenceChunker`] (SRX-based language-aware sentence splitting). It intentionally has
//! only two dependencies (`srx`, `whatlang`) so it can be used from wasm targets and from
//! crates that don't want the full `kalosm-language` stack.

use std::ops::Range;

pub mod sentence;

pub use sentence::{DefaultSentenceChunker, SentenceChunker};

/// A strategy for chunking a document into smaller pieces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChunkStrategy {
    /// Split the document into paragraphs.
    Paragraph {
        /// The number of paragraphs to include in each chunk.
        paragraph_count: usize,
        /// The number of paragraphs to overlap between chunks.
        overlap: usize,
    },
    /// Split the document into sentences.
    Sentence {
        /// The number of sentences to include in each chunk.
        sentence_count: usize,
        /// The number of sentences to overlap between chunks.
        overlap: usize,
    },
    /// Split the document into words.
    Words {
        /// The number of words to include in each chunk.
        word_count: usize,
        /// The number of words to overlap between chunks.
        overlap: usize,
    },
}

impl ChunkStrategy {
    /// Chunk a string into smaller ranges.
    pub fn chunk_str(&self, string: &str) -> Vec<Range<usize>> {
        match self {
            Self::Paragraph {
                paragraph_count,
                overlap,
            } => {
                let mut chunks = Vec::new();
                let mut start = 0;
                let mut newline_indexes = Vec::new();
                for (i, c) in string.char_indices() {
                    if c == '\n' {
                        newline_indexes.push(i + 1);
                        if newline_indexes.len() >= *paragraph_count {
                            if !string[start..i].trim().is_empty() {
                                chunks.push(start..i);
                            }
                            for _ in 0..(newline_indexes.len() - *overlap) {
                                start = newline_indexes.remove(0);
                            }
                        }
                    }
                }

                if !string[start..].trim().is_empty() {
                    chunks.push(start..string.len());
                }

                chunks
            }
            Self::Sentence {
                sentence_count,
                overlap,
            } => {
                let mut chunks = Vec::new();

                let splits = SentenceChunker::default().split_sentences(string);

                for window in splits
                    .windows(*sentence_count)
                    .step_by(*sentence_count - *overlap)
                {
                    if window.len() < *sentence_count {
                        break;
                    }
                    let start = window.first().unwrap().start;
                    let end = window.last().unwrap().end;
                    chunks.push(start..end);
                }

                chunks
            }
            Self::Words {
                word_count,
                overlap,
            } => {
                let mut chunks = Vec::new();
                let mut start = 0;
                let mut word_start_indexes = Vec::new();
                for (i, c) in string.char_indices() {
                    if c == ' ' {
                        word_start_indexes.push(i + 1);
                        if word_start_indexes.len() >= *word_count {
                            if !string[start..i].trim().is_empty() {
                                chunks.push(start..i);
                            }
                            for _ in 0..(word_start_indexes.len() - *overlap) {
                                start = word_start_indexes.remove(0);
                            }
                        }
                    }
                }

                if !string[start..].trim().is_empty() {
                    chunks.push(start..string.len());
                }

                chunks
            }
        }
    }
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        Self::Paragraph {
            paragraph_count: 3,
            overlap: 1,
        }
    }
}

#[test]
fn test_chunking() {
    let string = "The quick brown fox jumps over the lazy dog.";
    let chunks = ChunkStrategy::Words {
        word_count: 3,
        overlap: 1,
    };
    let chunks = chunks.chunk_str(string);
    assert_eq!(chunks.len(), 4);
    assert_eq!(string[chunks[0].clone()].trim(), "The quick brown");
    assert_eq!(string[chunks[1].clone()].trim(), "brown fox jumps");
    assert_eq!(string[chunks[2].clone()].trim(), "jumps over the");
    assert_eq!(string[chunks[3].clone()].trim(), "the lazy dog.");

    let chunks = ChunkStrategy::Sentence {
        sentence_count: 2,
        overlap: 1,
    };
    let string = "first sentence. second sentence. third sentence. fourth sentence.";
    let chunks = chunks.chunk_str(string);
    assert_eq!(chunks.len(), 3);
}
