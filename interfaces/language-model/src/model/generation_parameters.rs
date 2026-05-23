/// Parameters to use when generating text.
#[derive(Debug)]
pub struct GenerationParameters {
    pub(crate) temperature: f32,
    pub(crate) tau: f32,
    pub(crate) eta: f32,
    pub(crate) mu: f32,
    pub(crate) top_p: Option<f64>,
    pub(crate) top_k: Option<u32>,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) repetition_penalty_range: u32,
    pub(crate) max_length: u32,
    pub(crate) stop_on: Option<String>,
    pub(crate) seed: Option<u64>,
}

impl PartialEq for GenerationParameters {
    fn eq(&self, other: &Self) -> bool {
        self.temperature == other.temperature
            && self.eta == other.eta
            && self.tau == other.tau
            && self.mu == other.mu
            && self.top_p == other.top_p
            && self.top_k == other.top_k
            && self.repetition_penalty == other.repetition_penalty
            && self.repetition_penalty_range == other.repetition_penalty_range
            && self.max_length == other.max_length
            && self.stop_on == other.stop_on
    }
}

impl Clone for GenerationParameters {
    fn clone(&self) -> Self {
        Self {
            temperature: self.temperature,
            eta: self.eta,
            tau: self.tau,
            mu: self.mu,
            top_p: self.top_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            repetition_penalty_range: self.repetition_penalty_range,
            max_length: self.max_length,
            stop_on: self.stop_on.clone(),
            seed: None,
        }
    }
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationParameters {
    /// Create a new [`GenerationParameters`]
    pub const fn new() -> Self {
        Self {
            temperature: 0.8,
            eta: 0.1,
            tau: 5.,
            mu: 10.,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            repetition_penalty_range: 64,
            max_length: u32::MAX,
            stop_on: None,
            seed: None,
        }
    }

    /// Set the top_p parameter to the generation parameters (only used by the OpenAI API).
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set the top_k parameter to the generation parameters.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set the temperature to use when generating text.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the tau to use when generating text.
    pub fn with_tau(mut self, tau: f32) -> Self {
        self.tau = tau;
        self
    }

    /// Set the eta to use when generating text.
    pub fn with_eta(mut self, eta: f32) -> Self {
        self.eta = eta;
        self
    }

    /// Set the mu to use when generating text.
    pub fn with_mu(mut self, mu: f32) -> Self {
        self.mu = mu;
        self
    }

    /// Set the repetition penalty to use when generating text.
    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = Some(repetition_penalty);
        self
    }

    /// Set the repetition penalty range to use when generating text.
    pub fn with_repetition_penalty_range(mut self, repetition_penalty_range: u32) -> Self {
        self.repetition_penalty_range = repetition_penalty_range;
        self
    }

    /// Set the maximum length to use when generating text.
    pub fn with_max_length(mut self, max_length: u32) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set the string to stop on when generating text.
    pub fn with_stop_on(mut self, stop_on: impl Into<Option<String>>) -> Self {
        self.stop_on = stop_on.into();
        self
    }

    /// Set the seed to use when generating text.
    pub fn with_seed(mut self, seed: impl Into<Option<u64>>) -> Self {
        self.seed = seed.into();
        self
    }

    /// Get the temperature to use when generating text.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Get the tau to use when generating text.
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// Get the eta to use when generating text.
    pub fn eta(&self) -> f32 {
        self.eta
    }

    /// Get the mu to use when generating text.
    pub fn mu(&self) -> f32 {
        self.mu
    }

    /// Get the repetition penalty to use when generating text.
    pub fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty.unwrap_or(1.3)
    }

    /// Get the repetition penalty range to use when generating text.
    pub fn repetition_penalty_range(&self) -> u32 {
        self.repetition_penalty_range
    }

    /// Get the top-k sampling limit to use when generating text.
    pub fn top_k(&self) -> Option<u32> {
        self.top_k
    }

    /// Get the maximum length to use when generating text.
    pub fn max_length(&self) -> u32 {
        self.max_length
    }

    /// Get the string to stop on when generating text.
    pub fn stop_on(&self) -> Option<&str> {
        self.stop_on.as_deref()
    }

    /// Get the seed to use when generating text.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }
}
