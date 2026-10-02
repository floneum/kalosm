//! Seeded xorshift32 for weight initialization, batch selection and sampling.

/// xorshift32.
#[derive(Clone)]
pub struct Rng(u32);

impl Rng {
    /// A generator for `seed`. The zero state is a fixed point of xorshift,
    /// so it is nudged off it.
    pub fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Uniform in `[-1, 1)`.
    pub fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    /// Uniform in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n.max(1)
    }

    /// Draw an index from `weights`, which are expected to sum to one.
    ///
    /// Float error can leave the cumulative sum a hair under the draw, so the
    /// last index is the fallback rather than a panic.
    pub fn pick(&mut self, weights: &[f32]) -> usize {
        let mut remaining = self.unit() * weights.iter().sum::<f32>();
        for (i, w) in weights.iter().enumerate() {
            remaining -= w;
            if remaining <= 0.0 {
                return i;
            }
        }
        weights.len().saturating_sub(1)
    }
}
