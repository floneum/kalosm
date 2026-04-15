use super::{LlamaConfig, RopeScalingConfig};
use fusor::{
    arange, base_inverse_frequency, CastTensor, CastTo, DataType, Device, FloatDataType,
    SimdElement, Tensor,
};
use std::f32::consts::PI;

pub(crate) fn create_inverse_frequency<F>(
    rope_scaling: Option<&RopeScalingConfig>,
    rope_freq_weight: Option<&Tensor<1, F>>,
    dim: usize,
    rope_theta: f32,
    device: &Device,
) -> Tensor<2, F>
where
    F: FloatDataType + SimdElement + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    let mut inverse_frequency = base_inverse_frequency(dim, rope_theta);
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
    let inverse_frequency_len = inverse_frequency.len();
    let mut inverse_frequency_f32: Tensor<2, f32> = Tensor::new(device, &inverse_frequency)
        .reshape([1, inverse_frequency_len])
        .to_concrete();
    if let Some(weight) = &rope_freq_weight {
        let weight_f32: Tensor<1, f32> = weight.cast();
        inverse_frequency_f32 =
            (inverse_frequency_f32 * weight_f32.reshape((1, ())).to_concrete()).to_concrete();
    }

    inverse_frequency_f32.cast()
}

#[derive(Clone)]
pub(crate) enum RopeImplementation<F: FloatDataType + SimdElement = f32> {
    QwenVL(QwenVLRopeCache<F>),
    Llama(fusor::RopeCache),
}

impl<F: FloatDataType + SimdElement> RopeImplementation<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    pub fn new(config: &LlamaConfig<F>, rope_theta: f32, device: &Device) -> fusor::Result<Self> {
        if let Some(mrope_sections) = &config.mrope_sections {
            let cache = QwenVLRopeCache::new(config, rope_theta, mrope_sections, device)?;
            Ok(Self::QwenVL(cache))
        } else {
            let inverse_frequency: Tensor<2, F> = create_inverse_frequency(
                config.rope_scaling.as_ref(),
                config.rope_freq_weight.as_ref(),
                config.head_dimension,
                rope_theta,
                device,
            );
            let inverse_frequency_f32: Tensor<2, f32> = inverse_frequency.cast();
            let context_indices: Tensor<2, f32> =
                arange(device, 0f32, config.context_length as f32)
                    .reshape([config.context_length, 1])
                    .to_concrete();
            let outer_product = context_indices.mat_mul(&inverse_frequency_f32);
            let sin = outer_product.sin().to_concrete();
            let cos = outer_product.cos().to_concrete();
            Ok(Self::Llama(fusor::RopeCache::from_parts(cos, sin)))
        }
    }

    pub fn forward(
        &self,
        query: &Tensor<4, F>,
        key: &Tensor<4, F>,
        start_pos: usize,
        position_ids: Option<&Tensor<2, F>>,
        interleaved_rope: bool,
    ) -> (Tensor<4, F>, Tensor<4, F>) {
        match self {
            Self::QwenVL(cache) => cache.forward(
                position_ids.expect("qwen vl requires position ids"),
                query,
                key,
            ),
            Self::Llama(cache) => {
                let q_f32: Tensor<4, f32> = query.cast();
                let k_f32: Tensor<4, f32> = key.cast();
                let (q_out, k_out) = if interleaved_rope {
                    cache.forward_interleaved(&q_f32, &k_f32, start_pos)
                } else {
                    cache.forward(&q_f32, &k_f32, start_pos)
                };
                (q_out.cast(), k_out.cast())
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct QwenVLRopeCache<F: FloatDataType + SimdElement = f32> {
    inverse_frequency: Tensor<2, F>,
    mrope_sections: Vec<usize>,
}

impl<F: FloatDataType + SimdElement> QwenVLRopeCache<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    pub fn new(
        config: &LlamaConfig<F>,
        rope_theta: f32,
        mrope_sections: &[usize],
        device: &Device,
    ) -> fusor::Result<Self> {
        let inverse_frequency = create_inverse_frequency(
            config.rope_scaling.as_ref(),
            config.rope_freq_weight.as_ref(),
            config.head_dimension,
            rope_theta,
            device,
        );
        let mrope_sections = mrope_sections.iter().copied().filter(|&x| x > 0).collect();
        Ok(Self {
            inverse_frequency,
            mrope_sections,
        })
    }

    fn forward_sin_cos(&self, position_ids: &Tensor<2, F>) -> (Tensor<2, f32>, Tensor<2, f32>) {
        // Work in f32 for SIMD compatibility
        let inv_freq_f32: Tensor<2, f32> = self.inverse_frequency.cast();
        let position_ids_f32: Tensor<2, f32> = position_ids.cast();

        let inv_freq_expanded = inv_freq_f32
            .reshape(((),))
            .repeat([3])
            .reshape((3, 1, (), 1))
            .to_concrete();
        let position_ids_expanded = position_ids_f32.unsqueeze(1).unsqueeze(1).to_concrete();
        let freqs = inv_freq_expanded
            .mat_mul(&position_ids_expanded)
            .transpose(2, 3)
            .to_concrete();
        let cos = freqs.cos().to_concrete();
        let sin = freqs.sin().to_concrete();

        // Resolve dimension for cat
        // cos/sin are 4D: [3, batch, seq, head_dim]
        // After i(m, (i % 3, .., .., ..)), result is 3D: [batch, seq, split_size]
        // So we cat on dimension 2 (the last dimension of the 3D result)
        let last_dim_4d = cos.shape().len() - 1; // dimension 3 for splitting the 4D tensor
        let last_dim_3d = last_dim_4d - 1; // dimension 2 for concatenating the 3D results

        let cos = Tensor::cat(
            split(&cos, last_dim_4d, &self.mrope_sections)
                .iter()
                .enumerate()
                .map(|(i, m)| Tensor::<4, f32>::i(m, (i % 3, .., .., ..)).to_concrete())
                .collect::<Vec<_>>(),
            last_dim_3d,
        )
        .squeeze(0)
        .to_concrete();
        let sin = Tensor::cat(
            split(&sin, last_dim_4d, &self.mrope_sections)
                .iter()
                .enumerate()
                .map(|(i, m)| Tensor::<4, f32>::i(m, (i % 3, .., .., ..)).to_concrete())
                .collect::<Vec<_>>(),
            last_dim_3d,
        )
        .squeeze(0)
        .to_concrete();

        (cos, sin)
    }

    pub(crate) fn forward(
        &self,
        position_ids: &Tensor<2, F>,
        query: &Tensor<4, F>,
        key: &Tensor<4, F>,
    ) -> (Tensor<4, F>, Tensor<4, F>) {
        let (cos, sin) = self.forward_sin_cos(position_ids);
        // Rope operations work in f32, then cast back
        let query_f32: Tensor<4, f32> = query.cast();
        let key_f32: Tensor<4, f32> = key.cast();
        let key_out = key_f32.rope(&cos, &sin);
        let query_out = query_f32.rope(&cos, &sin);
        (query_out.cast(), key_out.cast())
    }
}

fn split<const R: usize, T: DataType + SimdElement>(
    tensor: &Tensor<R, T>,
    dim: usize,
    split_at: &[usize],
) -> Vec<Tensor<R, T>> {
    let mut result = Vec::new();
    let mut start = 0;
    for len in split_at.iter().copied() {
        let slice = tensor.narrow(dim, start, len).to_concrete();
        result.push(slice);
        start += len;
    }
    result
}

#[cfg(test)]
#[tokio::test]
async fn test_rope_cache() {
    use fusor::{Device, RopeCache, Tensor};

    let config: LlamaConfig<f32> = LlamaConfig::mock_test();
    let device = Device::new().await.unwrap();
    let cache = RopeCache::new(
        config.head_dimension,
        config.context_length,
        config.rope_theta,
        &device,
    )
    .unwrap();

    let expected_cos: Tensor<2, f32> = Tensor::new(
        &device,
        &[
            1.0000f32, 0.5403f32, -0.4161f32, -0.9900f32, -0.6536f32, 0.2837f32,
        ],
    )
    .reshape([6, 1])
    .to_concrete();
    let expected_sin: Tensor<2, f32> = Tensor::new(
        &device,
        &[
            0.0000f32, 0.8415f32, 0.9093f32, 0.1411f32, -0.7568f32, -0.9589f32,
        ],
    )
    .reshape([6, 1])
    .to_concrete();

    let cos_error: f32 = (cache.cos().clone() - expected_cos)
        .abs()
        .sum(0)
        .sum(0)
        .to_scalar()
        .await
        .unwrap();
    assert!(cos_error < 1e-2);
    let sin_error: f32 = (cache.sin().clone() - expected_sin)
        .abs()
        .sum(0)
        .sum(0)
        .to_scalar()
        .await
        .unwrap();
    assert!(sin_error < 1e-2);
}
