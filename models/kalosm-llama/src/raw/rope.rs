use super::{LlamaConfig, RopeScalingConfig};
use fusor2::composite::rope::{
    rope_normal_pair_fused, rope_normal_pair_fused_with_position, rope_pair_fused,
    rope_pair_fused_with_position,
};
use fusor2::device::Device;
use fusor2::graph::Graph;
use fusor2::tensor::Tensor;
use fusor2::{Dtype, Result};
use fusor2::Dim;
use std::f32::consts::PI;

/// The base `1 / theta^(2i/dim)` frequencies with the llama3-style scaling and
/// the optional per-frequency GGUF weights (`rope_freqs.weight`) applied.
pub(crate) fn create_inverse_frequency(
    rope_scaling: Option<&RopeScalingConfig>,
    rope_freq_weight: Option<&[f32]>,
    dim: usize,
    rope_theta: f32,
) -> Vec<f32> {
    let mut inverse_frequency =
        fusor2::composite::rope::base_inverse_frequency(dim as u32, rope_theta);
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

/// A `[context, head_dim / 2]` sin/cos table over custom inverse frequencies,
/// consumed by `fusor2::composite::rope`.
#[derive(Clone)]
pub(crate) struct RopeImplementation {
    cos: Tensor,
    sin: Tensor,
}

impl RopeImplementation {
    pub fn new(config: &LlamaConfig, rope_theta: f32, device: &Device) -> Result<Self> {
        let inverse_frequency = create_inverse_frequency(
            config.rope_scaling.as_ref(),
            config.rope_freq_weight.as_deref(),
            config.head_dimension,
            rope_theta,
        );
        Self::from_inverse_frequency(&inverse_frequency, config.context_length, device.graph())
    }

    pub(crate) fn from_inverse_frequency(
        inverse_frequency: &[f32],
        context_length: usize,
        graph: &Graph,
    ) -> Result<Self> {
        let half = inverse_frequency.len();
        let mut sin = Vec::with_capacity(context_length * half * 4);
        let mut cos = Vec::with_capacity(context_length * half * 4);
        for pos in 0..context_length {
            for f in inverse_frequency {
                // Accumulate the angle in f64: at large positions an f32
                // product has already lost the low bits.
                let angle = pos as f64 * *f as f64;
                sin.extend_from_slice(&(angle.sin() as f32).to_le_bytes());
                cos.extend_from_slice(&(angle.cos() as f32).to_le_bytes());
            }
        }
        let shape = [Dim::Const(context_length as u64), Dim::Const(half as u64)];
        Ok(Self {
            sin: graph.tensor(Dtype::F32, &shape, &sin)?,
            cos: graph.tensor(Dtype::F32, &shape, &cos)?,
        })
    }

    /// Rotate `q` and `k` at `start_pos`. `interleaved` pairs `(2i, 2i+1)`
    /// (the classic llama layout); otherwise halves `(i, i + Dh/2)`.
    ///
    /// `positions` is the decode-loop form: a rank-1 `u32` position tensor
    /// whose *bytes* change per step, so one graph serves every step. The
    /// host offset is a fallback for position-less callers.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        start_pos: usize,
        interleaved: bool,
        positions: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        match (positions, interleaved) {
            (Some(p), true) => rope_pair_fused_with_position(query, key, &self.cos, &self.sin, p),
            (Some(p), false) => {
                rope_normal_pair_fused_with_position(query, key, &self.cos, &self.sin, p)
            }
            (None, true) => rope_pair_fused(query, key, &self.cos, &self.sin, start_pos as u64),
            (None, false) => {
                rope_normal_pair_fused(query, key, &self.cos, &self.sin, start_pos as u64)
            }
        }
    }
}
