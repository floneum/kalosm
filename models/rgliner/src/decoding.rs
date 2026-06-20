//! Entity decoding with flat and nested NER support.

/// A recognized named entity.
#[derive(Debug, Clone)]
pub struct Entity {
    /// The entity text span.
    pub text: String,
    /// The entity label/type.
    pub label: String,
    /// Start character offset in the original text.
    pub start_char: usize,
    /// End character offset in the original text (exclusive).
    pub end_char: usize,
    /// Start word index.
    pub start_word: usize,
    /// End word index (inclusive).
    pub end_word: usize,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
}

/// Decoding mode for NER.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DecodingMode {
    /// Flat NER: no overlapping entities allowed.
    /// Uses greedy non-maximum suppression.
    #[default]
    Flat,
    /// Nested NER: overlapping entities allowed if one fully contains the other.
    /// Partial overlaps are still forbidden.
    Nested,
}

/// Entity decoder using non-maximum suppression.
pub struct Decoder {
    /// Confidence threshold for entity detection.
    threshold: f32,
    /// Decoding mode (flat or nested).
    mode: DecodingMode,
}

impl Decoder {
    /// Create a new decoder with the given threshold and mode.
    pub fn new(threshold: f32, mode: DecodingMode) -> Self {
        Self { threshold, mode }
    }

    /// Set the confidence threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the decoding mode.
    pub fn with_mode(mut self, mode: DecodingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Decode entity predictions for a single text.
    ///
    /// # Arguments
    /// * `scores` - Score matrix [num_spans, num_labels] after sigmoid
    /// * `span_indices` - (start_word, end_word) for each span
    /// * `word_offsets` - (start_char, end_char) for each word
    /// * `labels` - Label strings
    /// * `text` - Original input text
    pub fn decode(
        &self,
        scores: &[f32],
        num_spans: usize,
        num_labels: usize,
        span_indices: &[(usize, usize)],
        word_offsets: &[(usize, usize)],
        labels: &[&str],
        text: &str,
    ) -> Vec<Entity> {
        // Collect all predictions above threshold
        let mut candidates: Vec<(usize, usize, f32)> = Vec::new(); // (span_idx, label_idx, score)

        for span_idx in 0..num_spans {
            for label_idx in 0..num_labels {
                let score = scores[span_idx * num_labels + label_idx];
                if score >= self.threshold {
                    candidates.push((span_idx, label_idx, score));
                }
            }
        }

        // Sort by score descending
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        match self.mode {
            DecodingMode::Flat => {
                self.decode_flat(candidates, span_indices, word_offsets, labels, text)
            }
            DecodingMode::Nested => {
                self.decode_nested(candidates, span_indices, word_offsets, labels, text)
            }
        }
    }

    fn decode_flat(
        &self,
        candidates: Vec<(usize, usize, f32)>,
        span_indices: &[(usize, usize)],
        word_offsets: &[(usize, usize)],
        labels: &[&str],
        text: &str,
    ) -> Vec<Entity> {
        let num_words = word_offsets.len();
        let mut entities = Vec::new();
        let mut used_positions: Vec<bool> = vec![false; num_words];

        for (span_idx, label_idx, score) in candidates {
            let (start_word, end_word) = span_indices[span_idx];

            // Check if any word in span is already used
            let overlaps =
                (start_word..=end_word).any(|w| used_positions.get(w).copied().unwrap_or(false));

            if !overlaps {
                // Mark words as used
                for w in start_word..=end_word {
                    if w < used_positions.len() {
                        used_positions[w] = true;
                    }
                }

                if let Some(entity) = self.create_entity(
                    start_word,
                    end_word,
                    label_idx,
                    score,
                    word_offsets,
                    labels,
                    text,
                ) {
                    entities.push(entity);
                }
            }
        }

        // Sort by position
        entities.sort_by_key(|e| e.start_char);
        entities
    }

    fn decode_nested(
        &self,
        candidates: Vec<(usize, usize, f32)>,
        span_indices: &[(usize, usize)],
        word_offsets: &[(usize, usize)],
        labels: &[&str],
        text: &str,
    ) -> Vec<Entity> {
        let mut entities = Vec::new();
        // Track selected (start, end, label) triples to avoid exact duplicates
        let mut selected: Vec<(usize, usize, usize)> = Vec::new();

        for (span_idx, label_idx, score) in candidates {
            let (start_word, end_word) = span_indices[span_idx];
            let key = (start_word, end_word, label_idx);

            // Check for partial overlap with already selected entities
            let has_partial_overlap = selected.iter().any(|(sel_start, sel_end, _)| {
                self.is_partial_overlap(start_word, end_word, *sel_start, *sel_end)
            });

            // For nested NER, allow if no partial overlap and not exact duplicate
            if !has_partial_overlap && !selected.contains(&key) {
                selected.push(key);

                if let Some(entity) = self.create_entity(
                    start_word,
                    end_word,
                    label_idx,
                    score,
                    word_offsets,
                    labels,
                    text,
                ) {
                    entities.push(entity);
                }
            }
        }

        // Sort by position, then by span length descending (outer spans first)
        entities.sort_by(|a, b| {
            a.start_char
                .cmp(&b.start_char)
                .then_with(|| b.end_char.cmp(&a.end_char))
        });
        entities
    }

    /// Check if two spans have partial overlap (overlap but neither contains the other).
    fn is_partial_overlap(&self, start1: usize, end1: usize, start2: usize, end2: usize) -> bool {
        // Check if spans overlap at all
        let overlaps = start1 <= end2 && start2 <= end1;
        if !overlaps {
            return false;
        }

        // Check if one fully contains the other (not partial overlap)
        let one_contains_other =
            (start1 <= start2 && end1 >= end2) || (start2 <= start1 && end2 >= end1);

        overlaps && !one_contains_other
    }

    fn create_entity(
        &self,
        start_word: usize,
        end_word: usize,
        label_idx: usize,
        score: f32,
        word_offsets: &[(usize, usize)],
        labels: &[&str],
        text: &str,
    ) -> Option<Entity> {
        if start_word >= word_offsets.len() || end_word >= word_offsets.len() {
            return None;
        }
        if label_idx >= labels.len() {
            return None;
        }

        let start_char = word_offsets[start_word].0;
        let end_char = word_offsets[end_word].1;

        if end_char > text.len() {
            return None;
        }

        Some(Entity {
            text: text[start_char..end_char].to_string(),
            label: labels[label_idx].to_string(),
            start_char,
            end_char,
            start_word,
            end_word,
            score,
        })
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            mode: DecodingMode::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "Barack Obama visited Hawaii";
    // word -> (start_char, end_char): Barack, Obama, visited, Hawaii
    fn word_offsets() -> Vec<(usize, usize)> {
        vec![(0, 6), (7, 12), (13, 20), (21, 27)]
    }

    /// Flatten `(span_idx, label_idx, score)` triples into the row-major
    /// `[num_spans * num_labels]` score buffer `decode` expects.
    fn scores(num_spans: usize, num_labels: usize, entries: &[(usize, usize, f32)]) -> Vec<f32> {
        let mut s = vec![0.0; num_spans * num_labels];
        for &(span, label, score) in entries {
            s[span * num_labels + label] = score;
        }
        s
    }

    #[test]
    fn flat_suppresses_overlapping_lower_scoring_span() {
        // span 0 = "Barack Obama" (words 0..=1), span 1 = "Barack" (0..=0),
        // span 2 = "Hawaii" (3..=3). Labels: 0=person, 1=location.
        let span_indices = [(0, 1), (0, 0), (3, 3)];
        let labels = ["person", "location"];
        let s = scores(
            3,
            2,
            &[
                (0, 0, 0.9), // "Barack Obama" / person
                (1, 0, 0.6), // "Barack" / person  (overlaps span 0)
                (2, 1, 0.8), // "Hawaii" / location
            ],
        );

        let entities = Decoder::new(0.5, DecodingMode::Flat).decode(
            &s,
            3,
            2,
            &span_indices,
            &word_offsets(),
            &labels,
            TEXT,
        );

        // span 1 shares word 0 with the higher-scoring span 0 -> suppressed.
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "Barack Obama");
        assert_eq!(entities[0].label, "person");
        assert_eq!((entities[0].start_char, entities[0].end_char), (0, 12));
        assert_eq!(entities[1].text, "Hawaii");
        assert_eq!(entities[1].label, "location");
        assert_eq!((entities[1].start_char, entities[1].end_char), (21, 27));
    }

    #[test]
    fn nested_keeps_contained_span_but_drops_partial_overlap() {
        // span 0 = (0..=1), span 1 = (0..=0) fully contained in span 0,
        // span 2 = (1..=2) partially overlaps span 0.
        let span_indices = [(0, 1), (0, 0), (1, 2)];
        let labels = ["person"];
        let s = scores(3, 1, &[(0, 0, 0.9), (1, 0, 0.7), (2, 0, 0.6)]);

        let entities = Decoder::new(0.5, DecodingMode::Nested).decode(
            &s,
            3,
            1,
            &span_indices,
            &word_offsets(),
            &labels,
            TEXT,
        );

        // Contained span (0..=0) is allowed; partial overlap (1..=2) is not.
        assert_eq!(entities.len(), 2);
        let spans: Vec<(usize, usize)> = entities
            .iter()
            .map(|e| (e.start_word, e.end_word))
            .collect();
        assert!(spans.contains(&(0, 1)));
        assert!(spans.contains(&(0, 0)));
        assert!(!spans.contains(&(1, 2)));
    }

    #[test]
    fn scores_below_threshold_are_dropped() {
        let span_indices = [(0, 0)];
        let labels = ["person"];
        let s = scores(1, 1, &[(0, 0, 0.49)]);
        let entities = Decoder::new(0.5, DecodingMode::Flat).decode(
            &s,
            1,
            1,
            &span_indices,
            &word_offsets(),
            &labels,
            TEXT,
        );
        assert!(entities.is_empty());
    }

    #[test]
    fn is_partial_overlap_predicate() {
        let d = Decoder::default();
        // Disjoint.
        assert!(!d.is_partial_overlap(0, 1, 2, 3));
        // Adjacent ranges sharing word 1, neither containing the other -> partial.
        assert!(d.is_partial_overlap(0, 1, 1, 2));
        // Containment is not "partial".
        assert!(!d.is_partial_overlap(0, 3, 1, 2));
        assert!(!d.is_partial_overlap(1, 2, 0, 3));
        // Genuine partial overlap.
        assert!(d.is_partial_overlap(0, 2, 1, 3));
        // Identical spans -> containment, not partial.
        assert!(!d.is_partial_overlap(1, 2, 1, 2));
    }
}
