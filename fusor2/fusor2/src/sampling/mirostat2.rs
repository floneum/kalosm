//! Mirostat v2: surprise-targeting sampling with a running mu.

use crate::Result;
use crate::tensor::Tensor;

use super::row;
use super::top_k::GpuSampledToken;

/// Mirostat v2.
///
/// Each draw truncates the sorted distribution at the first token whose
/// surprise `-log2(p)` exceeds `mu`, samples from what is left, and then moves
/// `mu` against the surprise it actually observed:
/// `mu <- mu - eta * (surprise - tau)`.
pub struct Mirostat2Sampler {
    pub tau: f32,
    pub eta: f32,
    /// The running surprise target. Starts at `2 * tau`.
    pub mu: f32,
    pub seed: u64,
    /// How many draws this sampler has made. Mixed into `seed` so successive
    /// draws differ while the whole sequence stays a function of the seed.
    pub step: u64,
}

impl Mirostat2Sampler {
    pub fn new(tau: f32, eta: f32) -> Self {
        Self {
            tau,
            eta,
            mu: 2.0 * tau,
            seed: 0,
            step: 0,
        }
    }

    /// Draw one token and advance `mu`.
    ///
    /// The token stays on the device. `mu`, however, is a host `f32` on this
    /// struct, so the updated value has to be read back — one four-byte sync
    /// per draw. The logits themselves never leave the device.
    pub fn sample(&mut self, logits: &Tensor) -> Result<GpuSampledToken> {
        let n = row::row_len(logits)?;
        let graph = logits.graph().clone();

        let (sorted_values, sorted_ids) = row::sort_desc(logits, n)?;
        let weights = row::weights_of(&sorted_values, n)?;
        let total = row::fanout(&row::total_of(&weights)?, n)?.max_scalar(row::EPSILON)?;

        // surprise[r] = -log2(p[r]); non-increasing weights make it
        // non-decreasing, so the tokens failing the mu test are a suffix and
        // keeping the ones that pass is a prefix cutoff.
        let probability = weights.div(&total)?.max_scalar(row::EPSILON)?;
        let surprise = probability.log2()?.neg()?;
        let within = surprise.lte_scalar(self.mu)?;
        // The cutoff is clamped to at least one candidate.
        let keep = within.maximum(&row::first_only(&graph, n)?)?;
        let masked = weights.mul(&keep)?;

        let random = row::unit_random_at(self.seed, self.step);
        let one_hot = row::pick_one_hot(&masked, n, random)?;
        let token = row::gather_one_hot(&one_hot, &sorted_ids)?;

        // The observed surprise is taken against the truncated mass, which is
        // what `weighted_pick` divides by.
        let cutoff_sum = row::total_of(&masked)?.max_scalar(row::EPSILON)?;
        let picked = row::gather_one_hot(&one_hot, &weights)?;
        let observed = picked
            .div(&cutoff_sum)?
            .max_scalar(row::EPSILON)?
            .log2()?
            .neg()?;
        let next_mu = observed
            .sub_scalar(self.tau)?
            .mul_scalar(self.eta)?
            .rsub_scalar(self.mu)?;

        let token = row::as_token(&token)?;
        row::remember(&graph, token.id());

        // Resolve only the new mu. This forces the draw to be computed, but
        // nothing wider than one f32 crosses back.
        self.mu = next_mu.to_vec_f32()?.first().copied().unwrap_or(self.mu);
        self.step = self.step.wrapping_add(1);

        Ok(GpuSampledToken { value: token })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::test_support::{conformance_row, cpu_row};

    #[test]
    fn mu_starts_at_twice_tau() {
        let s = Mirostat2Sampler::new(3.0, 0.5);
        assert_eq!(s.mu, 6.0);
    }

    #[test]
    fn every_draw_is_in_the_vocabulary() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        let mut sampler = Mirostat2Sampler::new(5.0, 0.1);
        for _ in 0..8 {
            let got = sampler.sample(&t).unwrap().to_u32().unwrap();
            assert!((got as usize) < values.len(), "drew {got}");
        }
    }

    /// mu tracks the observed surprise across draws.
    #[test]
    fn mu_moves_with_the_observed_surprise() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        let mut sampler = Mirostat2Sampler::new(3.0, 0.5);
        let start = sampler.mu;
        for _ in 0..8 {
            sampler.sample(&t).unwrap();
        }
        assert!((sampler.mu - start).abs() > 1e-6, "mu never left {start}");
    }

    /// The update is `mu <- mu - eta * (surprise - tau)`; with a
    /// one-token-dominant row the surprise is ~0, so mu climbs by about
    /// `eta * tau` on the first draw.
    #[test]
    fn the_mu_update_follows_the_reference_recurrence() {
        // A row so peaked that the truncated set is a single token, whose
        // renormalised probability is 1 and whose surprise is 0.
        let mut values = vec![-40.0f32; 8];
        values[5] = 40.0;
        let (_s, _g, t) = cpu_row(&values);
        let mut sampler = Mirostat2Sampler::new(2.0, 0.25);
        let before = sampler.mu;
        let token = sampler.sample(&t).unwrap().to_u32().unwrap();
        assert_eq!(token, 5, "the only plausible token");
        // surprise = 0 => mu <- mu - eta * (0 - tau) = mu + eta * tau.
        let want = before + 0.25 * 2.0;
        assert!(
            (sampler.mu - want).abs() < 1e-3,
            "mu moved to {}, want {want}",
            sampler.mu
        );
    }

    /// mu falls when the observed surprise is above tau. Eight equal logits
    /// give every token a renormalised probability of 1/8 and a surprise of
    /// exactly 3 bits: `4 - 0.5 * (3 - 2) = 3.5`.
    #[test]
    fn a_surprise_above_tau_pulls_mu_down() {
        let (_s, _g, t) = cpu_row(&[1.0f32; 8]);
        let mut sampler = Mirostat2Sampler::new(2.0, 0.5);
        assert_eq!(sampler.mu, 4.0);
        sampler.sample(&t).unwrap();
        assert!(
            (sampler.mu - 3.5).abs() < 1e-4,
            "mu moved to {}, want 3.5",
            sampler.mu
        );
    }

    /// The same seed replays the same sequence of draws and the same mu.
    #[test]
    fn the_sequence_is_seed_deterministic() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        let run = || {
            let mut sampler = Mirostat2Sampler::new(4.0, 0.2);
            sampler.seed = 1234;
            let draws: Vec<u32> = (0..6)
                .map(|_| sampler.sample(&t).unwrap().to_u32().unwrap())
                .collect();
            (draws, sampler.mu)
        };
        let (a, mu_a) = run();
        let (b, mu_b) = run();
        assert_eq!(a, b, "the same seed must replay");
        assert_eq!(mu_a, mu_b);
    }

    /// Successive draws from one sampler are not frozen on one token.
    #[test]
    fn successive_draws_advance_the_stream() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        let mut sampler = Mirostat2Sampler::new(8.0, 0.05);
        sampler.seed = 77;
        let draws: std::collections::BTreeSet<u32> = (0..32)
            .map(|_| sampler.sample(&t).unwrap().to_u32().unwrap())
            .collect();
        assert!(draws.len() > 1, "32 draws all returned {draws:?}");
    }

    #[test]
    fn the_pending_token_is_a_usable_device_operand() {
        let values = conformance_row();
        let (_s, _g, t) = cpu_row(&values);
        let mut sampler = Mirostat2Sampler::new(5.0, 0.1);
        let pending = sampler.sample(&t).unwrap();
        assert_eq!(pending.value.dtype(), crate::Dtype::U32);
        assert!(pending.value.rank() <= 1);
        assert!(pending.value.add_scalar(0u32).is_ok());
    }
}
