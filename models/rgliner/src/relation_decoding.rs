//! Relation types for GLiNER-RelEx.
//!
//! The relation decoding logic lives in [`crate::relex`], which scores every
//! directed entity pair against the relation labels and keeps those above the
//! relation threshold. This module just defines the public [`Relation`] result.

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
