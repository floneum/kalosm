//! Mirostat v2: surprise-targeting sampling with a running mu.

use crate::Result;
use crate::tensor::Tensor;

use super::row;
use super::top_k::GpuSampledToken;

/// Mirostat v2 sampler.
///
/// Each draw truncates the sorted distribution at the first token whose
/// surprise `-log2(p)` exceeds `mu`, samples from what is left, and then moves
/// `mu` against the surprise it actually observed:
/// `mu <- mu - eta * (surprise - tau)`.
pub struct Mirostat2Sampler {
    /// Target surprise.
    pub tau: f32,
    /// Adaptation rate.
    pub eta: f32,
    /// The running surprise target. Starts at `2 * tau`.
    pub mu: f32,
    /// Base random seed.
    pub seed: u64,
    /// How many draws this sampler has made. Mixed into `seed` so successive
    /// draws differ while the whole sequence stays a function of the seed.
    pub step: u64,
}

impl Mirostat2Sampler {
    /// Create a sampler with `mu = 2 * tau` and seed zero.
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
        let (token, next_mu) = self.draw(logits)?;
        // Resolve only the new mu. This forces the draw to be computed, but
        // nothing wider than one f32 crosses back.
        self.mu = next_mu.to_vec_f32()?.first().copied().unwrap_or(self.mu);
        self.step = self.step.wrapping_add(1);
        Ok(token)
    }

    /// [`Self::sample`], awaited: the one-float readback that updates `mu`
    /// is the only host sync, so this is the form a browser can use.
    pub async fn sample_async(&mut self, logits: &Tensor) -> Result<GpuSampledToken> {
        let (token, next_mu) = self.draw(logits)?;
        self.mu = next_mu
            .to_vec_f32_async()
            .await?
            .first()
            .copied()
            .unwrap_or(self.mu);
        self.step = self.step.wrapping_add(1);
        Ok(token)
    }

    /// Build the draw and the next `mu` on the device; nothing is resolved.
    fn draw(&self, logits: &Tensor) -> Result<(GpuSampledToken, Tensor)> {
        let n = row::row_len(logits)?;
        let graph = logits.graph().clone();

        let (sorted_values, sorted_ids) = row::sort_desc(logits, n)?;
        let weights = row::weights_of(&sorted_values, n)?;
        let total = row::fanout(&row::total_of(&weights)?, n)?.max_scalar(row::EPSILON)?;

        // surprise[r] = -log2(p[r]); non-increasing weights make it
        // non-decreasing, so the tokens failing the mu test are a suffix.
        let probability = weights.div(&total)?.max_scalar(row::EPSILON)?;
        let surprise = probability.log2()?.neg()?;
        let within = surprise.lte_scalar(self.mu)?;
        // Clamp the cutoff to at least one candidate.
        let keep = within.maximum(&row::first_only(&graph, n)?)?;
        let masked = weights.mul(&keep)?;

        let random = row::unit_random_at(self.seed, self.step);
        let one_hot = row::pick_one_hot(&masked, n, random)?;
        let token = row::gather_one_hot(&one_hot, &sorted_ids)?;

        // The observed surprise is taken against the *truncated* mass, which
        // is what `weighted_pick` divides by.
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
        Ok((GpuSampledToken { value: token }, next_mu))
    }
}
