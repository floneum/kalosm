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
    /// The original shape, retained for comparable compiler benchmarks.
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
        let embeddings = vocab * self.dim + self.context * self.dim;
        let attention = self.dim + 4 * self.dim * self.dim;
        let feed_forward = self.dim + 2 * self.dim * self.mlp;
        embeddings + self.blocks * (attention + feed_forward) + self.dim + self.dim * vocab
    }

    pub fn validate(self, vocab: usize) -> Result<(), String> {
        for (name, value, min, max) in [
            ("Blocks", self.blocks, 1, 8),
            ("Model width", self.dim, 8, 512),
            ("Attention heads", self.heads, 1, 16),
            ("Feed-forward width", self.mlp, 8, 2048),
            ("Context", self.context, 8, 512),
            ("Batch size", self.batch, 1, 128),
            ("Vocabulary", vocab, 2, 128),
        ] {
            if !(min..=max).contains(&value) {
                return Err(format!("{name} must be between {min} and {max}."));
            }
        }
        if !self.dim.is_multiple_of(self.heads) {
            return Err("Model width must be divisible by attention heads.".into());
        }
        // A conservative admission budget, not a promise about device memory.
        // Account for parameters, Adam, gradients and retained activations;
        // attention grows quadratically with context. Check dimensions first
        // so even hostile integer inputs cannot overflow this arithmetic.
        let working_bytes = 4
            * (12 * self.parameters(vocab) as u64
                + self.blocks as u64
                    * self.tokens() as u64
                    * (16 * self.dim + 4 * self.mlp + 6 * self.heads * self.context) as u64);
        if working_bytes > 256 * 1024 * 1024 {
            return Err("This combination exceeds the browser's 256 MiB working-memory budget. Reduce batch size, context or width.".into());
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
        if !(1..=1_000_000_000).contains(&self.token_budget) {
            return Err("Training tokens must be between 1 and 1,000,000,000.".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_invalid_and_excessive_shapes_before_allocating() {
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
                ..ModelConfig::TINY
            },
            ModelConfig {
                context: 512,
                batch: 128,
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
        let run = TrainingConfig {
            token_budget: 2049,
            ..Default::default()
        };
        assert_eq!(run.steps(), 2);
        assert_eq!(ModelConfig::TINY.parameters(65), 240_480);
    }
}
