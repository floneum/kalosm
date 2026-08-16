use super::{LlamaConfig, RopeScalingConfig};
use fusor::composite::base_inverse_frequency;
use fusor::{Device, Tensor};
use std::f32::consts::PI;

/// The base `1 / theta^(2i/dim)` frequencies with the llama3-style scaling and
/// the optional per-frequency GGUF weights (`rope_freqs.weight`) applied.
pub(crate) fn create_inverse_frequency(
    rope_scaling: Option<&RopeScalingConfig>,
    rope_freq_weight: Option<&[f32]>,
    dim: usize,
    rope_theta: f32,
) -> Vec<f32> {
    let mut inverse_frequency = base_inverse_frequency(dim as u32, rope_theta);
    if let Some(scaling_config) = &rope_scaling {
        let original_max_position_embeddings = scaling_config.original_max_position_embeddings;
        let factor = scaling_config.factor;
        let high_freq_factor = scaling_config.high_freq_factor;
        let low_freq_factor = scaling_config.low_freq_factor;
        let low_freq_wavelen = original_max_position_embeddings as f32 / low_freq_factor;
        let high_freq_wavelen = original_max_position_embeddings as f32 / high_freq_factor;
        for freq in inverse_frequency.iter_mut() {
            let wavelen = 2. * PI / *freq;
            if wavelen > low_freq_wavelen {
                *freq /= factor
            } else if wavelen == high_freq_wavelen {
                let smooth = (original_max_position_embeddings as f32 / wavelen - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                *freq = (1. - smooth) * *freq / factor + smooth * *freq
            }
        }
    }
    if let Some(weight) = rope_freq_weight {
        for (freq, w) in inverse_frequency.iter_mut().zip(weight.iter()) {
            *freq *= w;
        }
    }
    inverse_frequency
}

/// A `[context, head_dim / 2]` sin/cos table over custom inverse frequencies.
#[derive(Clone)]
pub(crate) struct RopeImplementation {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

impl RopeImplementation {
    pub fn new(config: &LlamaConfig, rope_theta: f32, device: &Device) -> Self {
        let inverse_frequency = create_inverse_frequency(
            config.rope_scaling.as_ref(),
            config.rope_freq_weight.as_deref(),
            config.head_dimension,
            rope_theta,
        );
        Self::from_inverse_frequency(&inverse_frequency, config.context_length, device)
    }

    pub(crate) fn from_inverse_frequency(
        inverse_frequency: &[f32],
        context_length: usize,
        device: &Device,
    ) -> Self {
        let half = inverse_frequency.len();
        let mut sin = Vec::with_capacity(context_length * half);
        let mut cos = Vec::with_capacity(context_length * half);
        for pos in 0..context_length {
            for f in inverse_frequency {
                // Accumulate the angle in f64: at large positions an f32
                // product has already lost the low bits.
                let angle = pos as f64 * *f as f64;
                sin.push(angle.sin() as f32);
                cos.push(angle.cos() as f32);
            }
        }
        let shape = [context_length, half];
        Self {
            sin: Tensor::from_slice(device, shape, &sin),
            cos: Tensor::from_slice(device, shape, &cos),
        }
    }

    /// Rotate `q` and `k` at `start_pos`. `interleaved` pairs `(2i, 2i+1)`
    /// (the classic llama layout); otherwise halves `(i, i + Dh/2)`.
    ///
    /// `positions` is the decode-loop form: a rank-1 `u32` position tensor
    /// whose *bytes* change per step, so one graph serves every step. The
    /// host offset is a fallback for position-less callers.
    pub fn forward(
        &self,
        query: &Tensor<4>,
        key: &Tensor<4>,
        start_pos: usize,
        interleaved: bool,
        positions: Option<&Tensor<1, u32>>,
    ) -> (Tensor<4>, Tensor<4>) {
        let (cos, sin) = (&self.cos, &self.sin);
        match (positions, interleaved) {
            (Some(p), true) => query.rope_interleaved_pair_at(key, cos, sin, p),
            (Some(p), false) => query.rope_pair_at(key, cos, sin, p),
            (None, true) => query.rope_interleaved_pair(key, cos, sin, start_pos as u64),
            (None, false) => query.rope_pair(key, cos, sin, start_pos as u64),
        }
    }
}
