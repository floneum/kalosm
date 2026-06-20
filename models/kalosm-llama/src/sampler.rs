use kalosm_language_model::SamplingStrategy;
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

pub(crate) enum CpuSampler {
    Mirostat2(CpuMirostat2Sampler),
    Standard(CpuStandardSampler),
}

impl CpuSampler {
    pub(crate) fn new(config: GpuSamplerConfig, seed: Option<u64>) -> Self {
        match config.sampling_strategy {
            SamplingStrategy::Mirostat2 => Self::Mirostat2(CpuMirostat2Sampler::new(config, seed)),
            SamplingStrategy::Standard => Self::Standard(CpuStandardSampler::new(config, seed)),
        }
    }

    pub(crate) fn sample_token(
        &mut self,
        logits: &mut [Logit],
        previous_tokens: &[u32],
        top_k: usize,
    ) -> Option<u32> {
        match self {
            Self::Mirostat2(sampler) => sampler.sample_token(logits, previous_tokens, top_k),
            Self::Standard(sampler) => sampler.sample_token(logits, previous_tokens, top_k),
        }
    }
}

pub(crate) struct CpuMirostat2Sampler {
    config: GpuSamplerConfig,
    mu: f32,
    rng: rand::rngs::StdRng,
}

impl CpuMirostat2Sampler {
    pub(crate) fn new(config: GpuSamplerConfig, seed: Option<u64>) -> Self {
        Self {
            mu: config.mu,
            config,
            rng: rng_from_seed(seed),
        }
    }

    pub(crate) fn sample_token(
        &mut self,
        logits: &mut [Logit],
        previous_tokens: &[u32],
        top_k: usize,
    ) -> Option<u32> {
        let top_k = top_k.max(1);
        let previous_tokens = previous_tokens_for_config(&self.config, previous_tokens);
        let mut top = logits
            .iter()
            .copied()
            .filter_map(|logit| process_logit(&self.config, logit, previous_tokens))
            .collect::<Vec<_>>();

        if top.is_empty() {
            return None;
        }

        sort_logits_desc(&mut top);
        top.truncate(top_k.min(top.len()));

        let max = top[0].logit;
        let total = top
            .iter()
            .map(|logit| (logit.logit - max).exp())
            .sum::<f32>()
            .max(EPSILON);

        let cutoff = self.candidate_cutoff(&top, max, total);

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

    fn candidate_cutoff(&self, top: &[Logit], max: f32, total: f32) -> usize {
        let mut cutoff = top.len();
        for (index, logit) in top.iter().enumerate() {
            let probability = ((logit.logit - max).exp() / total).max(EPSILON);
            if -probability.log2() > self.mu {
                cutoff = index.max(1);
                break;
            }
        }
        cutoff
    }
}

pub(crate) struct CpuStandardSampler {
    config: GpuSamplerConfig,
    rng: rand::rngs::StdRng,
}

impl CpuStandardSampler {
    pub(crate) fn new(config: GpuSamplerConfig, seed: Option<u64>) -> Self {
        Self {
            config,
            rng: rng_from_seed(seed),
        }
    }

    pub(crate) fn sample_token(
        &mut self,
        logits: &mut [Logit],
        previous_tokens: &[u32],
        top_k: usize,
    ) -> Option<u32> {
        let top_k = top_k.max(1);
        let previous_tokens = previous_tokens_for_config(&self.config, previous_tokens);
        let mut top = logits
            .iter()
            .copied()
            .filter_map(|logit| process_logit(&self.config, logit, previous_tokens))
            .collect::<Vec<_>>();

        if top.is_empty() {
            return None;
        }

        sort_logits_desc(&mut top);
        top.truncate(top_k.min(top.len()));

        if self.config.temperature <= 0.0 {
            return Some(top[0].token_id);
        }

        let max = top[0].logit;
        let mut candidates = top
            .into_iter()
            .map(|logit| WeightedLogit {
                logit,
                weight: (logit.logit - max).exp(),
            })
            .collect::<Vec<_>>();

        self.apply_min_p(&mut candidates);
        self.apply_top_p(&mut candidates);

        let total = candidates
            .iter()
            .map(|candidate| candidate.weight)
            .sum::<f32>()
            .max(EPSILON);
        let threshold = self.rng.random::<f32>().clamp(0.0, RANDOM_MAX) * total;
        let mut cumulative = 0.0;
        for candidate in &candidates {
            cumulative += candidate.weight;
            if cumulative >= threshold {
                return Some(candidate.logit.token_id);
            }
        }

        candidates.last().map(|candidate| candidate.logit.token_id)
    }

    fn apply_min_p(&self, candidates: &mut Vec<WeightedLogit>) {
        let Some(min_p) = self.config.min_p else {
            return;
        };
        let min_p = normalized_probability(min_p, 0.0);
        if min_p <= 0.0 {
            return;
        }
        let max_weight = candidates
            .first()
            .map(|candidate| candidate.weight)
            .unwrap_or(0.0);
        let threshold = max_weight * min_p;
        candidates.retain(|candidate| candidate.weight >= threshold);
    }

    fn apply_top_p(&self, candidates: &mut Vec<WeightedLogit>) {
        let Some(top_p) = self.config.top_p else {
            return;
        };
        let top_p = normalized_probability(top_p, 1.0);
        if top_p >= 1.0 {
            return;
        }
        let total = candidates
            .iter()
            .map(|candidate| candidate.weight)
            .sum::<f32>()
            .max(EPSILON);
        let threshold = total * top_p;
        let mut cumulative = 0.0;
        let mut cutoff = 0;
        for (index, candidate) in candidates.iter().enumerate() {
            cumulative += candidate.weight;
            cutoff = index + 1;
            if cumulative >= threshold {
                break;
            }
        }
        candidates.truncate(cutoff.max(1));
    }
}

#[derive(Clone, Copy, Debug)]
struct WeightedLogit {
    logit: Logit,
    weight: f32,
}

fn rng_from_seed(seed: Option<u64>) -> rand::rngs::StdRng {
    seed.map(rand::rngs::StdRng::seed_from_u64)
        .unwrap_or_else(rand::rngs::StdRng::from_os_rng)
}

fn sort_logits_desc(logits: &mut [Logit]) {
    logits.sort_unstable_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| right.token_id.cmp(&left.token_id))
    });
}

fn process_logit(
    config: &GpuSamplerConfig,
    mut logit: Logit,
    previous_tokens: &[u32],
) -> Option<Logit> {
    let mut value = logit.logit;
    if !value.is_finite() {
        return None;
    }

    let repetition_penalty = config.repetition_penalty;
    if repetition_penalty > 1.0 && previous_tokens.contains(&logit.token_id) {
        if value <= 0.0 {
            value *= repetition_penalty;
        } else {
            value /= repetition_penalty;
        }
    }

    let temperature = config.temperature;
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

fn previous_tokens_for_config<'a>(
    config: &GpuSamplerConfig,
    previous_tokens: &'a [u32],
) -> &'a [u32] {
    let len = previous_tokens.len().min(config.repetition_penalty_range);
    &previous_tokens[previous_tokens.len().saturating_sub(len)..]
}

fn normalized_probability(value: f32, default: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(mu: f32) -> GpuSamplerConfig {
        GpuSamplerConfig::new(1.0, 5.0, 0.1, mu, 1.0, 64, Some(4))
    }

    fn standard_config() -> GpuSamplerConfig {
        let mut config = test_config(10.0);
        config.sampling_strategy = SamplingStrategy::Standard;
        config
    }

    fn flat_logits() -> [Logit; 4] {
        [
            Logit {
                token_id: 3,
                logit: 0.0,
                prob: 0.0,
            },
            Logit {
                token_id: 2,
                logit: 0.0,
                prob: 0.0,
            },
            Logit {
                token_id: 1,
                logit: 0.0,
                prob: 0.0,
            },
            Logit {
                token_id: 0,
                logit: 0.0,
                prob: 0.0,
            },
        ]
    }

    #[test]
    fn candidate_cutoff_uses_all_top_k_when_surprise_never_exceeds_mu() {
        let sampler = CpuMirostat2Sampler::new(test_config(10.0), Some(0));
        let top = flat_logits();

        assert_eq!(sampler.candidate_cutoff(&top, 0.0, 4.0), top.len());
    }

    #[test]
    fn sample_token_can_choose_past_first_candidate_with_flat_logits() {
        let mut sampled = std::collections::BTreeSet::new();
        for seed in 0..64 {
            let mut sampler = CpuMirostat2Sampler::new(test_config(10.0), Some(seed));
            let mut logits = flat_logits();
            let top_k = logits.len();
            sampled.insert(sampler.sample_token(&mut logits, &[], top_k).unwrap());
        }

        assert!(
            sampled.iter().any(|token| *token != 3),
            "flat logits should not always return the first sorted candidate; sampled {sampled:?}"
        );
    }

    #[test]
    fn standard_sampler_can_choose_past_first_candidate_with_flat_logits() {
        let config = standard_config();
        let mut sampled = std::collections::BTreeSet::new();
        for seed in 0..64 {
            let mut sampler = CpuSampler::new(config, Some(seed));
            let mut logits = flat_logits();
            let top_k = logits.len();
            sampled.insert(sampler.sample_token(&mut logits, &[], top_k).unwrap());
        }

        assert!(
            sampled.iter().any(|token| *token != 3),
            "standard sampler should not always return the first sorted candidate; sampled {sampled:?}"
        );
    }

    #[test]
    fn standard_sampler_top_p_limits_candidate_set() {
        let mut config = standard_config();
        config.top_p = Some(0.5);
        let mut sampled = std::collections::BTreeSet::new();

        for seed in 0..64 {
            let mut sampler = CpuSampler::new(config, Some(seed));
            let mut logits = flat_logits();
            let top_k = logits.len();
            sampled.insert(sampler.sample_token(&mut logits, &[], top_k).unwrap());
        }

        assert!(
            sampled.iter().all(|token| matches!(*token, 2 | 3)),
            "top-p should restrict flat logits to the first two candidates; sampled {sampled:?}"
        );
    }

    #[test]
    fn standard_sampler_min_p_limits_candidate_set() {
        let mut config = standard_config();
        config.min_p = Some(0.1);
        let mut sampler = CpuSampler::new(config, Some(0));
        let mut logits = [
            Logit {
                token_id: 3,
                logit: 0.0,
                prob: 0.0,
            },
            Logit {
                token_id: 2,
                logit: -3.0,
                prob: 0.0,
            },
            Logit {
                token_id: 1,
                logit: -4.0,
                prob: 0.0,
            },
            Logit {
                token_id: 0,
                logit: -5.0,
                prob: 0.0,
            },
        ];
        let top_k = logits.len();

        assert_eq!(sampler.sample_token(&mut logits, &[], top_k), Some(3));
    }
}
