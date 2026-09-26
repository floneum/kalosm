//! Shape and run configuration, validated before requesting a GPU or building a graph.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelConfig {
    pub blocks: usize,
    pub dim: usize,
    pub heads: usize,
    pub mlp: usize,
    pub context: usize,
    pub batch: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            context: 128,
            ..Self::TINY
        }
    }
}

impl ModelConfig {
    /// The 64-token benchmark preset.
    pub const TINY: Self = Self {
        blocks: 3,
        dim: 96,
        heads: 4,
        mlp: 192,
        context: 64,
        batch: 16,
    };

    pub const fn tokens(self) -> usize {
        self.batch * self.context
    }
    pub const fn head_dim(self) -> usize {
        self.dim / self.heads
    }

    pub fn parameters(self, vocab: usize) -> usize {
        self.checked_parameters(vocab)
            .expect("model configuration must be checked before counting parameters")
    }

    fn checked_parameters(self, vocab: usize) -> Option<usize> {
        let embeddings = vocab.checked_add(self.context)?.checked_mul(self.dim)?;
        let attention = self
            .dim
            .checked_mul(self.dim)?
            .checked_mul(4)?
            .checked_add(self.dim)?;
        let feed_forward = self
            .dim
            .checked_mul(self.mlp)?
            .checked_mul(2)?
            .checked_add(self.dim)?;
        embeddings
            .checked_add(
                self.blocks
                    .checked_mul(attention.checked_add(feed_forward)?)?,
            )?
            .checked_add(self.dim)?
            .checked_add(self.dim.checked_mul(vocab)?)
    }

    pub fn validate(self, vocab: usize) -> Result<(), String> {
        for (name, value) in [
            ("Blocks", self.blocks),
            ("Model width", self.dim),
            ("Attention heads", self.heads),
            ("Feed-forward width", self.mlp),
            ("Context", self.context),
            ("Batch size", self.batch),
        ] {
            if value == 0 {
                return Err(format!("{name} must be a positive whole number."));
            }
        }
        // Token ids are bytes, independently of model size.
        if !(2..=256).contains(&vocab) {
            return Err("The vocabulary must contain between 2 and 256 tokens.".into());
        }
        if !self.dim.is_multiple_of(self.heads) {
            return Err("Model width must be divisible by attention heads.".into());
        }
        self.checked_parameters(vocab)
            .ok_or("The parameter count exceeds this platform's address space.")?;
        // Every tensor must fit a host allocation; the backend enforces device limits.
        for (name, shape) in [
            ("Embedding", &[vocab, self.dim][..]),
            ("Position", &[self.context, self.dim][..]),
            ("Attention projection", &[self.dim, self.dim][..]),
            ("Feed-forward projection", &[self.dim, self.mlp][..]),
            ("Hidden", &[self.batch, self.context, self.dim][..]),
            ("Feed-forward", &[self.batch, self.context, self.mlp][..]),
            (
                "Attention",
                &[self.batch, self.heads, self.context, self.context][..],
            ),
            ("Logits", &[self.batch, self.context, vocab][..]),
        ] {
            let bytes = shape
                .iter()
                .try_fold(size_of::<f32>(), |n, d| n.checked_mul(*d));
            if bytes.is_none_or(|n| n > isize::MAX as usize) {
                return Err(format!(
                    "The {name} tensor exceeds this platform's address space."
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub token_budget: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            token_budget: 20_000_000,
        }
    }
}

impl TrainingConfig {
    /// Complete batches; the run may exceed the requested budget by <1 step.
    pub fn steps(self) -> u64 {
        self.token_budget.div_ceil(self.model.tokens() as u64)
    }

    pub fn validate(self, vocab: usize) -> Result<(), String> {
        self.model.validate(vocab)?;
        if self.token_budget == 0 {
            return Err("Training tokens must be a positive whole number.".into());
        }
        self.steps()
            .checked_mul(self.model.tokens() as u64)
            .ok_or("The token budget rounded to complete batches exceeds the token counter.")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_invalid_and_unrepresentable_shapes_before_allocating() {
        for bad in [
            ModelConfig {
                heads: 0,
                ..ModelConfig::TINY
            },
            ModelConfig {
                heads: 5,
                ..ModelConfig::TINY
            },
            ModelConfig {
                dim: usize::MAX,
                heads: 1,
                ..ModelConfig::TINY
            },
            ModelConfig {
                context: usize::MAX / 4,
                batch: 8,
                ..ModelConfig::TINY
            },
        ] {
            assert!(bad.validate(65).is_err(), "{bad:?}");
        }
        assert!(TrainingConfig::default().validate(96).is_ok());
        assert!(
            TrainingConfig {
                token_budget: 0,
                ..Default::default()
            }
            .validate(65)
            .is_err()
        );
        assert!(
            TrainingConfig {
                token_budget: u64::MAX,
                ..Default::default()
            }
            .validate(96)
            .is_err()
        );
    }
}
