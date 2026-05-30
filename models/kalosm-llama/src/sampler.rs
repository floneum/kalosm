use rand::{Rng, SeedableRng};

use crate::GpuSamplerConfig;

const EPSILON: f32 = 1.0e-20;
const RANDOM_MAX: f32 = 0.999_999_94;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Logit {
    pub(crate) token_id: u32,
    pub(crate) logit: f32,
    pub(crate) prob: f32,
}

pub(crate) type Logits = Vec<Logit>;

pub(crate) struct CpuMirostat2Sampler {
    config: GpuSamplerConfig,
    mu: f32,
    rng: rand::rngs::StdRng,
}

impl CpuMirostat2Sampler {
    pub(crate) fn new(config: GpuSamplerConfig, seed: Option<u64>) -> Self {
        let rng = seed
            .map(rand::rngs::StdRng::seed_from_u64)
            .unwrap_or_else(rand::rngs::StdRng::from_os_rng);
        Self {
            mu: config.mu,
            config,
            rng,
        }
    }

    pub(crate) fn sample_token(
        &mut self,
        logits: &mut [Logit],
        previous_tokens: &[u32],
        top_k: usize,
    ) -> Option<u32> {
        let top_k = top_k.max(1);
        let previous_tokens = self.previous_tokens(previous_tokens);
        let mut top = logits
            .iter()
            .copied()
            .filter_map(|logit| self.process_logit(logit, previous_tokens))
            .collect::<Vec<_>>();

        if top.is_empty() {
            return None;
        }

        top.sort_unstable_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| right.token_id.cmp(&left.token_id))
        });
        top.truncate(top_k.min(top.len()));

        let max = top[0].logit;
        let total = top
            .iter()
            .map(|logit| (logit.logit - max).exp())
            .sum::<f32>()
            .max(EPSILON);

        let mut cutoff = 0;
        for (index, logit) in top.iter().enumerate() {
            let probability = ((logit.logit - max).exp() / total).max(EPSILON);
            if -probability.log2() > self.mu {
                cutoff = index.max(1);
                break;
            }
        }
        if cutoff == 0 {
            cutoff = 1;
        }

        let candidates = &top[..cutoff.min(top.len())];
        let cutoff_sum = candidates
            .iter()
            .map(|logit| (logit.logit - max).exp())
            .sum::<f32>()
            .max(EPSILON);
        let threshold = self.rng.random::<f32>().clamp(0.0, RANDOM_MAX) * cutoff_sum;
        let mut cumulative = 0.0;
        let mut selected = candidates[0];
        let mut selected_weight = (selected.logit - max).exp();

        for logit in candidates {
            let weight = (logit.logit - max).exp();
            cumulative += weight;
            if cumulative >= threshold {
                selected = *logit;
                selected_weight = weight;
                break;
            }
        }

        let selected_probability = (selected_weight / cutoff_sum).max(EPSILON);
        let surprise = -selected_probability.log2();
        let next_mu = self.mu - self.config.eta * (surprise - self.config.tau);
        if next_mu.is_finite() {
            self.mu = next_mu;
        }

        Some(selected.token_id)
    }

    fn process_logit(&self, mut logit: Logit, previous_tokens: &[u32]) -> Option<Logit> {
        let mut value = logit.logit;
        if !value.is_finite() {
            return None;
        }

        let repetition_penalty = self.config.repetition_penalty;
        if repetition_penalty > 1.0 && previous_tokens.contains(&logit.token_id) {
            if value <= 0.0 {
                value *= repetition_penalty;
            } else {
                value /= repetition_penalty;
            }
        }

        let temperature = self.config.temperature;
        if temperature != 0.0 {
            value /= temperature;
        }

        if !value.is_finite() {
            return None;
        }

        logit.logit = value;
        logit.prob = 0.0;
        Some(logit)
    }

    fn previous_tokens<'a>(&self, previous_tokens: &'a [u32]) -> &'a [u32] {
        let len = previous_tokens
            .len()
            .min(self.config.repetition_penalty_range);
        &previous_tokens[previous_tokens.len().saturating_sub(len)..]
    }
}
