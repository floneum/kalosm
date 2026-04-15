//! Relation decoding with three-threshold approach.
//!
//! Three thresholds control the extraction pipeline:
//! 1. Entity threshold: Confidence cutoff for NER
//! 2. Adjacency threshold: Entity pair candidate filtering
//! 3. Relation threshold: Relation classification cutoff

use crate::decoding::Entity;

/// A recognized relation between two entities.
#[derive(Debug, Clone)]
pub struct Relation {
    /// The head (source) entity.
    pub head: Entity,
    /// The tail (target) entity.
    pub tail: Entity,
    /// The relation type/label.
    pub relation: String,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
}

/// Configuration for relation decoding thresholds.
#[derive(Debug, Clone)]
pub struct RelationDecoderConfig {
    /// Entity detection threshold (default: 0.4)
    pub entity_threshold: f32,
    /// Adjacency filtering threshold (default: 0.55)
    pub adjacency_threshold: f32,
    /// Relation classification threshold (default: 0.8)
    pub relation_threshold: f32,
}

impl Default for RelationDecoderConfig {
    fn default() -> Self {
        Self {
            entity_threshold: 0.4,
            adjacency_threshold: 0.55,
            relation_threshold: 0.8,
        }
    }
}

/// Relation decoder using three-threshold approach.
pub struct RelationDecoder {
    config: RelationDecoderConfig,
}

impl RelationDecoder {
    /// Create a new relation decoder with default config.
    pub fn new() -> Self {
        Self {
            config: RelationDecoderConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: RelationDecoderConfig) -> Self {
        Self { config }
    }

    /// Set entity threshold.
    pub fn with_entity_threshold(mut self, threshold: f32) -> Self {
        self.config.entity_threshold = threshold;
        self
    }

    /// Set adjacency threshold.
    pub fn with_adjacency_threshold(mut self, threshold: f32) -> Self {
        self.config.adjacency_threshold = threshold;
        self
    }

    /// Set relation threshold.
    pub fn with_relation_threshold(mut self, threshold: f32) -> Self {
        self.config.relation_threshold = threshold;
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &RelationDecoderConfig {
        &self.config
    }

    /// Decode relations from entity pairs and relation scores.
    ///
    /// # Arguments
    /// * `entities` - Detected entities from NER stage
    /// * `adjacency_scores` - Adjacency matrix scores [num_entities, num_entities]
    /// * `relation_scores` - Relation scores for each pair [num_pairs, num_relations]
    /// * `candidate_pairs` - (head_idx, tail_idx) pairs that passed adjacency threshold
    /// * `relation_labels` - Relation type labels
    ///
    /// # Returns
    /// Vector of decoded relations above the relation threshold
    pub fn decode(
        &self,
        entities: &[Entity],
        _adjacency_scores: &[f32],
        relation_scores: &[f32],
        candidate_pairs: &[(usize, usize)],
        relation_labels: &[&str],
    ) -> Vec<Relation> {
        let num_relations = relation_labels.len();
        let mut relations = Vec::new();

        for (pair_idx, &(head_idx, tail_idx)) in candidate_pairs.iter().enumerate() {
            if head_idx >= entities.len() || tail_idx >= entities.len() {
                continue;
            }

            // Get relation scores for this pair
            let scores_start = pair_idx * num_relations;
            let scores_end = scores_start + num_relations;

            if scores_end > relation_scores.len() {
                continue;
            }

            // Find relations above threshold
            for (rel_idx, &score) in relation_scores[scores_start..scores_end].iter().enumerate() {
                if score >= self.config.relation_threshold {
                    relations.push(Relation {
                        head: entities[head_idx].clone(),
                        tail: entities[tail_idx].clone(),
                        relation: relation_labels[rel_idx].to_string(),
                        score,
                    });
                }
            }
        }

        // Sort by score descending
        relations.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        relations
    }

    /// Filter entity pairs based on adjacency scores.
    ///
    /// # Arguments
    /// * `adjacency_scores` - Flat adjacency matrix [num_entities * num_entities]
    /// * `num_entities` - Number of entities
    ///
    /// # Returns
    /// Vector of (head_idx, tail_idx) pairs above the adjacency threshold
    pub fn filter_pairs(
        &self,
        adjacency_scores: &[f32],
        num_entities: usize,
    ) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        for i in 0..num_entities {
            for j in 0..num_entities {
                if i == j {
                    continue; // Skip self-relations
                }

                let idx = i * num_entities + j;
                if idx < adjacency_scores.len() {
                    let score = adjacency_scores[idx];
                    if score >= self.config.adjacency_threshold {
                        pairs.push((i, j));
                    }
                }
            }
        }

        pairs
    }

    /// Pool entity span embeddings by mean pooling.
    ///
    /// # Arguments
    /// * `text_embeddings` - Text token embeddings [num_words, hidden_size]
    /// * `entity` - Entity with word indices
    ///
    /// # Returns
    /// Mean-pooled embedding for the entity span
    pub fn pool_entity_embedding(
        text_embeddings: &[f32],
        hidden_size: usize,
        entity: &Entity,
    ) -> Vec<f32> {
        let start = entity.start_word;
        let end = entity.end_word;
        let num_words = end - start + 1;

        if num_words == 0 {
            return vec![0.0; hidden_size];
        }

        let mut pooled = vec![0.0f32; hidden_size];

        for word_idx in start..=end {
            let offset = word_idx * hidden_size;
            for h in 0..hidden_size {
                if offset + h < text_embeddings.len() {
                    pooled[h] += text_embeddings[offset + h];
                }
            }
        }

        // Divide by number of words for mean pooling
        for h in 0..hidden_size {
            pooled[h] /= num_words as f32;
        }

        pooled
    }
}

impl Default for RelationDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(start: usize, end: usize, label: &str) -> Entity {
        Entity {
            text: format!("entity_{start}_{end}"),
            label: label.to_string(),
            start_char: start * 5,
            end_char: end * 5 + 5,
            start_word: start,
            end_word: end,
            score: 0.9,
        }
    }

    #[test]
    fn test_filter_pairs() {
        let decoder = RelationDecoder::new().with_adjacency_threshold(0.5);

        let adjacency_scores = vec![
            0.0, 0.8, 0.3, // entity 0 -> [0, 1, 2]
            0.7, 0.0, 0.6, // entity 1 -> [0, 1, 2]
            0.2, 0.4, 0.0, // entity 2 -> [0, 1, 2]
        ];

        let pairs = decoder.filter_pairs(&adjacency_scores, 3);

        // Should include (0,1), (1,0), (1,2) where score >= 0.5
        assert!(pairs.contains(&(0, 1))); // 0.8
        assert!(pairs.contains(&(1, 0))); // 0.7
        assert!(pairs.contains(&(1, 2))); // 0.6
        assert!(!pairs.contains(&(0, 2))); // 0.3 < 0.5
    }

    #[test]
    fn test_decode_relations() {
        let decoder = RelationDecoder::new().with_relation_threshold(0.7);

        let entities = vec![
            make_entity(0, 1, "organization"),
            make_entity(3, 4, "person"),
            make_entity(6, 6, "location"),
        ];

        let adjacency_scores = vec![0.9; 9]; // All pairs pass
        let candidate_pairs = vec![(0, 1), (0, 2), (1, 2)];

        // Relation scores: 3 pairs x 2 relations
        let relation_scores = vec![
            0.85, 0.3, // pair (0,1): "founded by" = 0.85, "located in" = 0.3
            0.2, 0.9, // pair (0,2): "founded by" = 0.2, "located in" = 0.9
            0.1, 0.4, // pair (1,2): both below threshold
        ];

        let relation_labels = &["founded by", "located in"];

        let relations = decoder.decode(
            &entities,
            &adjacency_scores,
            &relation_scores,
            &candidate_pairs,
            relation_labels,
        );

        assert_eq!(relations.len(), 2);

        // Check first relation (highest score should be located_in at 0.9)
        assert_eq!(relations[0].relation, "located in");
        assert_eq!(relations[0].score, 0.9);

        // Check second relation
        assert_eq!(relations[1].relation, "founded by");
        assert_eq!(relations[1].score, 0.85);
    }
}
