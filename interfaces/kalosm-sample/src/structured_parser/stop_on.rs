use crate::{CreateParserState, ParseStatus, Parser};

type CharFilter = fn(char) -> bool;

/// A parser that parses until a literal is found.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct StopOn<S: AsRef<str> = &'static str, F: Fn(char) -> bool + 'static = CharFilter> {
    literal: S,
    character_filter: F,
    len_range: std::ops::RangeInclusive<usize>,
}

impl<S: AsRef<str>> CreateParserState for StopOn<S> {
    fn create_parser_state(&self) -> <Self as Parser>::PartialState {
        StopOnOffset::default()
    }
}

impl<S: AsRef<str>> From<S> for StopOn<S> {
    fn from(literal: S) -> Self {
        Self {
            literal,
            character_filter: |_| true,
            len_range: 0..=usize::MAX,
        }
    }
}

impl<S: AsRef<str>> StopOn<S> {
    /// Create a new literal parser.
    pub fn new(literal: S) -> Self {
        Self {
            literal,
            character_filter: |_| true,
            len_range: 0..=usize::MAX,
        }
    }
}

impl<S: AsRef<str>, F: Fn(char) -> bool + 'static> StopOn<S, F> {
    /// Only allow characters that pass the filter.
    pub fn filter_characters(self, character_filter: F) -> StopOn<S, F> {
        StopOn {
            literal: self.literal,
            character_filter,
            len_range: self.len_range,
        }
    }

    /// Set the length range of the parsed text (excluding the stop literal).
    pub fn with_length(mut self, len_range: std::ops::RangeInclusive<usize>) -> Self {
        self.len_range = len_range;
        self
    }

    /// Get the literal that this parser stops on.
    pub fn literal(&self) -> &str {
        self.literal.as_ref()
    }
}

/// An error that can occur while parsing a string literal.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct StopOnParseError;

impl std::fmt::Display for StopOnParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        "StopOnParseError".fmt(f)
    }
}

impl std::error::Error for StopOnParseError {}

/// The state of a stop on literal parser.
#[derive(Default, Debug, PartialEq, Eq, Clone)]
pub struct StopOnOffset {
    offset: usize,
    text: String,
}

impl StopOnOffset {
    /// Create a new stop on literal parser state.
    pub fn new(offset: usize) -> Self {
        Self {
            offset,
            text: String::new(),
        }
    }
}

impl<S: AsRef<str>, F: Fn(char) -> bool + 'static> Parser for StopOn<S, F> {
    type Output = String;
    type PartialState = StopOnOffset;

    fn parse<'a>(
        &self,
        state: &StopOnOffset,
        input: &'a [u8],
    ) -> crate::ParseResult<ParseStatus<'a, Self::PartialState, Self::Output>> {
        let mut new_offset = state.offset;
        let mut text = state.text.clone();

        let Ok(input_str) = std::str::from_utf8(input) else {
            crate::bail!(StopOnParseError);
        };
        let literal_length = self.literal.as_ref().len();
        let mut literal_iter = self.literal.as_ref()[state.offset..].chars();

        for (i, input_char) in input_str.char_indices() {
            if !(self.character_filter)(input_char) {
                crate::bail!(StopOnParseError);
            }

            let literal_char = literal_iter.next();

            if Some(input_char) == literal_char {
                new_offset += 1;

                if new_offset == literal_length {
                    text += std::str::from_utf8(&input[..i + 1]).unwrap();
                    // Check if the content length (excluding the stop literal) is within range
                    let content_len = text.len() - literal_length;
                    if !self.len_range.contains(&content_len) {
                        crate::bail!(StopOnParseError);
                    }
                    return Ok(ParseStatus::Finished {
                        result: text,
                        remaining: &input[i + 1..],
                    });
                }
            } else {
                literal_iter = self.literal.as_ref()[state.offset..].chars();
                new_offset = 0;
            }

            // Check if the content length (excluding partial literal match) exceeds max
            let current_content_len = text.len() + i + 1 - new_offset;
            if current_content_len > *self.len_range.end() {
                crate::bail!(StopOnParseError);
            }
        }

        text.push_str(input_str);

        Ok(ParseStatus::Incomplete {
            new_state: StopOnOffset {
                offset: new_offset,
                text,
            },
            required_next: "".into(),
        })
    }
}

#[test]
fn literal_parser() {
    let parser = StopOn::new("Hello, world!");
    let state = StopOnOffset {
        offset: 0,
        text: String::new(),
    };
    assert_eq!(
        parser.parse(&state, b"Hello, world!"),
        Ok(ParseStatus::Finished {
            result: "Hello, world!".to_string(),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, world! This is a test"),
        Ok(ParseStatus::Finished {
            result: "Hello, world!".to_string(),
            remaining: b" This is a test"
        })
    );
    assert_eq!(
        parser.parse(&state, b"Hello, "),
        Ok(ParseStatus::Incomplete {
            new_state: StopOnOffset {
                offset: 7,
                text: "Hello, ".into()
            },
            required_next: "".into()
        })
    );
    assert_eq!(
        parser.parse(
            &parser
                .parse(&state, b"Hello, ")
                .unwrap()
                .unwrap_incomplete()
                .0,
            b"world!"
        ),
        Ok(ParseStatus::Finished {
            result: "Hello, world!".to_string(),
            remaining: &[]
        })
    );
    assert_eq!(
        parser.parse(&state, b"Goodbye, world!"),
        Ok(ParseStatus::Incomplete {
            new_state: StopOnOffset {
                offset: 0,
                text: "Goodbye, world!".into()
            },
            required_next: "".into()
        })
    );
}

#[test]
fn stop_on_with_length() {
    // Test with max length of 5 (content before the stop literal)
    let parser = StopOn::new("!").with_length(0..=5);
    let state = StopOnOffset::default();

    // "Hello" is 5 chars, should succeed
    assert_eq!(
        parser.parse(&state, b"Hello!"),
        Ok(ParseStatus::Finished {
            result: "Hello!".to_string(),
            remaining: &[]
        })
    );

    // "Helloo" is 6 chars of content before "!", should fail
    assert!(parser.parse(&state, b"Helloo!").is_err());

    // Test min length
    let parser = StopOn::new("!").with_length(3..=10);
    // "Hi" is only 2 chars, should fail
    assert!(parser.parse(&state, b"Hi!").is_err());

    // "Hey" is 3 chars, should succeed
    assert_eq!(
        parser.parse(&state, b"Hey!"),
        Ok(ParseStatus::Finished {
            result: "Hey!".to_string(),
            remaining: &[]
        })
    );
}
