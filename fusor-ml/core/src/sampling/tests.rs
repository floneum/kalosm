use std::mem::size_of;

use crate::{DataTypeEnum, Device, Tensor, TensorData, quantized::QMatrix};
use fusor_gguf::{BlockQ4_0, GgmlType};

use crate::mir::kernel_backend::sampling::{
    ChunkProcessors, ProcessorSettings, chunk_top_k_pair_data_with_encoder,
    sample_categorical_logits_data_with_encoder, sample_from_sorted_top_k_data_with_encoder,
    supports_unfiltered_categorical,
};

use super::{
    GPU_SAMPLE_STATUS_INVALID, GPU_SAMPLE_STATUS_RETRY_NEEDED, GPU_SAMPLE_STATUS_SAMPLED,
    GpuMirostat2Sampler, GpuMirostat2SamplerParams, GpuSamplerRequest, GpuStandardSamplerParams,
    sample_token_to_host,
};

fn unfiltered_params(input_len: usize, temperature: f32, random: f32) -> GpuStandardSamplerParams {
    GpuStandardSamplerParams {
        top_k: input_len,
        temperature,
        repetition_penalty: 1.0,
        top_p: 1.0,
        min_p: 0.0,
        random,
    }
}

#[test]
fn top_k_pairs_match_cpu_sorted_order() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let values = [
            0.25,
            f32::NAN,
            7.0,
            -3.0,
            f32::INFINITY,
            2.5,
            9.0,
            f32::NEG_INFINITY,
            8.5,
            9.0,
            6.0,
            -1.0,
        ];
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(5).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(5);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    });
}

#[test]
fn processed_chunk_top_k_applies_temperature_and_repetition_penalty() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let values = [
            4.0,
            -2.0,
            3.5,
            8.0,
            f32::NAN,
            1.0,
            5.0,
            -1.5,
            6.5,
            7.0,
            f32::NEG_INFINITY,
            0.5,
        ];
        let buffer = device.create_buffer_init(
            bytemuck::cast_slice(&values),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let data = TensorData::new_from_buffer(&device, buffer, &[values.len()], DataTypeEnum::F32);
        let previous_tokens = [0, 3, 9];
        let (ids, logits) = chunk_top_k_pair_data_with_encoder(
            &data,
            Some(ChunkProcessors {
                previous_tokens: &previous_tokens,
                gpu_tail: None,
                settings: ProcessorSettings {
                    temperature: 0.5,
                    repetition_penalty: 2.0,
                },
            }),
            5,
            5,
            None,
        )
        .unwrap();
        let ids = Tensor::from(ids).as_slice::<1, u32>().await.unwrap();
        let logits = Tensor::from(logits).as_slice::<1, f32>().await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(token_id, mut value)| {
                if !value.is_finite() {
                    return None;
                }
                if previous_tokens.contains(&(token_id as u32)) {
                    if value <= 0.0 {
                        value *= 2.0;
                    } else {
                        value /= 2.0;
                    }
                }
                value /= 0.5;
                Some((token_id as u32, value))
            })
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(5);

        let actual = ids
            .as_slice()
            .iter()
            .copied()
            .zip(logits.as_slice().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    });
}

#[test]
fn unfiltered_categorical_fast_path_eligibility_is_strict() {
    let eligible = unfiltered_params(65, 0.8, 0.25);
    assert!(supports_unfiltered_categorical(65, eligible));

    let mut filtered = eligible;
    filtered.top_k = 64;
    assert!(!supports_unfiltered_categorical(65, filtered));
    filtered = eligible;
    filtered.top_p = 0.99;
    assert!(!supports_unfiltered_categorical(65, filtered));
    filtered = eligible;
    filtered.min_p = 0.01;
    assert!(!supports_unfiltered_categorical(65, filtered));
    filtered = eligible;
    filtered.temperature = f32::NAN;
    assert!(!supports_unfiltered_categorical(65, filtered));
    filtered = eligible;
    filtered.random = f32::NAN;
    assert!(!supports_unfiltered_categorical(65, filtered));
    assert!(!supports_unfiltered_categorical(0, eligible));
    assert!(!supports_unfiltered_categorical(257, eligible));
}

#[test]
fn backend_unfiltered_categorical_matches_generic_boundaries() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap().with_poisoned_allocations();
        let logits = vec![0.0; 65];
        let data = TensorData::new_from_buffer(
            &device,
            device.create_buffer_init(
                bytemuck::cast_slice(&logits),
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            &[logits.len()],
            DataTypeEnum::F32,
        );
        let generic_logits = Tensor::new(&device, logits.as_slice());
        let first_boundary = 1.0f32 / logits.len() as f32;
        let after_first_boundary = f32::from_bits(first_boundary.to_bits() + 1);
        let last_boundary = 64.0f32 / logits.len() as f32;
        for random in [
            0.0,
            first_boundary,
            after_first_boundary,
            0.5,
            last_boundary,
            0.999_999,
        ] {
            let params = unfiltered_params(logits.len(), 0.8, random);
            let expected = generic_logits
                .sample_standard_token(&[u32::MAX], params)
                .await
                .unwrap();
            let output = sample_categorical_logits_data_with_encoder(&data, params, None).unwrap();
            let result = Tensor::from(output).as_slice::<1, u32>().await.unwrap();
            assert_eq!(result.as_slice()[0], GPU_SAMPLE_STATUS_SAMPLED);
            assert_eq!(
                result.as_slice()[1],
                expected,
                "categorical boundary mismatch for random={random}"
            );
            if random == 0.0 {
                assert_eq!(expected, 64, "ties must prefer the larger token id");
            }
        }
    });
}

#[test]
fn unfiltered_categorical_matches_generic_seeded_order() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap().with_poisoned_allocations();
        let logits = [0.0, 2.0, 2.0, -1.0, 2.0, f32::NAN, 1.0];
        let tensor = Tensor::new(&device, logits.as_slice());

        for random in [0.0, 0.2, 0.5, 0.9, 0.999_999] {
            let params = unfiltered_params(logits.len(), 0.8, random);
            let fast = tensor.sample_standard_token(&[], params).await.unwrap();
            // A non-matching previous token forces the generic processed-top-k
            // route without changing any processed logits.
            let generic = tensor
                .sample_standard_token(&[u32::MAX], params)
                .await
                .unwrap();
            assert_eq!(fast, generic, "seeded sampler mismatch for random={random}");
            if random == 0.0 {
                assert_eq!(fast, 4, "ties must prefer the larger token id");
            }
        }
    });
}

#[test]
fn backend_unfiltered_categorical_handles_temperature_and_invalid_logits() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let logits = [f32::NAN, -3.0, 0.25, 2.0, f32::INFINITY, -1.0, 1.5];
        let data = TensorData::new_from_buffer(
            &device,
            device.create_buffer_init(
                bytemuck::cast_slice(&logits),
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            &[logits.len()],
            DataTypeEnum::F32,
        );
        let generic_logits = Tensor::new(&device, logits.as_slice());
        for (temperature, random) in [(0.5, 0.2), (0.5, 0.8), (2.0, 0.6)] {
            let params = unfiltered_params(logits.len(), temperature, random);
            let expected = generic_logits
                .sample_standard_token(&[u32::MAX], params)
                .await
                .unwrap();
            let output = sample_categorical_logits_data_with_encoder(&data, params, None).unwrap();
            let result = Tensor::from(output).as_slice::<1, u32>().await.unwrap();
            assert_eq!(
                result.as_slice(),
                &[GPU_SAMPLE_STATUS_SAMPLED, expected],
                "categorical mismatch for temperature={temperature} random={random}"
            );
        }

        let invalid_logits = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let invalid = TensorData::new_from_buffer(
            &device,
            device.create_buffer_init(
                bytemuck::cast_slice(&invalid_logits),
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            &[invalid_logits.len()],
            DataTypeEnum::F32,
        );
        let output = sample_categorical_logits_data_with_encoder(
            &invalid,
            unfiltered_params(invalid_logits.len(), 1.0, 0.5),
            None,
        )
        .unwrap();
        let result = Tensor::from(output).as_slice::<1, u32>().await.unwrap();
        assert_eq!(result.as_slice(), &[GPU_SAMPLE_STATUS_INVALID, 0]);
    });
}

#[test]
fn pending_unfiltered_categorical_exposes_the_sampled_gpu_token() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap().with_poisoned_allocations();
        let logits = vec![0.0; 65];
        let params = unfiltered_params(logits.len(), 0.8, 0.0);
        let expected = 64;
        let tensor = Tensor::new(&device, logits.as_slice()).sin();

        let pending = tensor
            .sample_standard_token_pending(&[], None, params)
            .expect("full-vocabulary categorical sampling should stay pending");
        let dependent = pending.token_tensor() + 7u32;
        let dependent = dependent.as_slice::<1, u32>().await.unwrap();
        assert_eq!(dependent.as_slice(), &[expected + 7]);
        assert_eq!(pending.read_token().await.unwrap(), Some(expected));
    });
}

fn cpu_mirostat2_selected_token(values: &[f32], mu: f32, params: GpuMirostat2SamplerParams) -> u32 {
    let mut top = values
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(token_id, value)| {
            value
                .is_finite()
                .then_some((token_id as u32, value / params.temperature))
        })
        .collect::<Vec<_>>();
    top.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| right.0.cmp(&left.0))
    });
    top.truncate(params.top_k.min(top.len()));

    let max_value = top[0].1;
    let total = top
        .iter()
        .map(|(_, value)| (*value - max_value).exp())
        .sum::<f32>()
        .max(1.0e-20);
    let mut cutoff = top.len();
    for (scan, (_, value)) in top.iter().enumerate() {
        let probability = (*value - max_value).exp() / total;
        if -probability.max(1.0e-20).log2() > mu {
            cutoff = scan.max(1);
            break;
        }
    }

    let cutoff_sum = top
        .iter()
        .take(cutoff)
        .map(|(_, value)| (*value - max_value).exp())
        .sum::<f32>()
        .max(1.0e-20);
    let threshold = params.random.clamp(0.0, 0.999_999_94) * cutoff_sum;
    let mut cumulative = 0.0;
    let mut selected = top[0].0;
    for (token_id, value) in top.iter().take(cutoff) {
        cumulative += (*value - max_value).exp();
        if cumulative >= threshold {
            selected = *token_id;
            break;
        }
    }
    selected
}

#[test]
fn tensor_mirostat2_sampler_uses_processed_top_k_order() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let values = [9.0, 8.5, 7.0, 6.0, 2.5, 0.25, -1.0, -3.0];
        for run_device in [
            device.clone(),
            device.without_subgroups(),
            device.with_poisoned_allocations(),
            device.without_subgroups().with_poisoned_allocations(),
        ] {
            let tensor = Tensor::new(&run_device, values.as_slice());
            let mut sampler = GpuMirostat2Sampler::new(&run_device, 10.0);
            let token = tensor
                .sample_mirostat2_token(
                    &mut sampler,
                    &[],
                    GpuMirostat2SamplerParams {
                        top_k: 4,
                        temperature: 1.0,
                        repetition_penalty: 1.0,
                        tau: 5.0,
                        eta: 0.1,
                        random: 0.0,
                    },
                )
                .await
                .unwrap();

            assert_eq!(token, 0);
        }
    });
}

#[test]
fn backend_mirostat2_sampler_uses_full_top_k_without_surprise_cutoff() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let values = [1.0, 1.0, 1.0, 1.0];
        let ids = [3u32, 2, 1, 0];
        let value_buffer = device.create_buffer_init(
            bytemuck::cast_slice(&values),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let id_buffer = device.create_buffer_init(
            bytemuck::cast_slice(&ids),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let values_data =
            TensorData::new_from_buffer(&device, value_buffer, &[values.len()], DataTypeEnum::F32);
        let ids_data =
            TensorData::new_from_buffer(&device, id_buffer, &[ids.len()], DataTypeEnum::U32);
        let mu = 10.0;
        let params = GpuMirostat2SamplerParams {
            top_k: values.len(),
            temperature: 1.0,
            repetition_penalty: 1.0,
            tau: 5.0,
            eta: 0.1,
            random: 0.8,
        };
        let mut sampler = GpuMirostat2Sampler::new(&device, mu);

        let output = sample_from_sorted_top_k_data_with_encoder(
            &ids_data,
            &values_data,
            &mut GpuSamplerRequest::Mirostat2 {
                sampler: &mut sampler,
                params,
            },
            None,
            None,
        )
        .unwrap();
        let result = Tensor::from(output).as_slice::<1, u32>().await.unwrap();

        assert_eq!(result.as_slice()[0], GPU_SAMPLE_STATUS_SAMPLED);
        assert_eq!(result.as_slice()[1], 0);
    });
}

#[test]
fn backend_standard_sampler_applies_top_p_on_gpu() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let values = [1.0, 1.0, 1.0, 1.0];
        let ids = [3u32, 2, 1, 0];
        let value_buffer = device.create_buffer_init(
            bytemuck::cast_slice(&values),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let id_buffer = device.create_buffer_init(
            bytemuck::cast_slice(&ids),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let values_data =
            TensorData::new_from_buffer(&device, value_buffer, &[values.len()], DataTypeEnum::F32);
        let ids_data =
            TensorData::new_from_buffer(&device, id_buffer, &[ids.len()], DataTypeEnum::U32);
        let params = GpuStandardSamplerParams {
            top_k: values.len(),
            temperature: 1.0,
            repetition_penalty: 1.0,
            top_p: 0.5,
            min_p: 0.0,
            random: 0.8,
        };

        let output = sample_from_sorted_top_k_data_with_encoder(
            &ids_data,
            &values_data,
            &mut GpuSamplerRequest::Standard { params },
            None,
            None,
        )
        .unwrap();
        let result = Tensor::from(output).as_slice::<1, u32>().await.unwrap();

        assert_eq!(result.as_slice()[0], GPU_SAMPLE_STATUS_SAMPLED);
        assert_eq!(result.as_slice()[1], 2);
    });
}

#[test]
fn backend_mirostat2_sampler_matches_cpu_reference_for_sorted_top_k() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let values = [9.0, 7.5, 7.0, 4.0, 3.5, 1.0];
        let ids = [0u32, 1, 2, 3, 4, 5];
        let value_buffer = device.create_buffer_init(
            bytemuck::cast_slice(&values),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let id_buffer = device.create_buffer_init(
            bytemuck::cast_slice(&ids),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let values_data =
            TensorData::new_from_buffer(&device, value_buffer, &[values.len()], DataTypeEnum::F32);
        let ids_data =
            TensorData::new_from_buffer(&device, id_buffer, &[ids.len()], DataTypeEnum::U32);
        let mu = 3.0;
        let params = GpuMirostat2SamplerParams {
            top_k: 5,
            temperature: 1.0,
            repetition_penalty: 1.0,
            tau: 5.0,
            eta: 0.1,
            random: 0.65,
        };
        let expected = cpu_mirostat2_selected_token(&values, mu, params);
        let mut sampler = GpuMirostat2Sampler::new(&device, mu);

        let output = sample_from_sorted_top_k_data_with_encoder(
            &ids_data,
            &values_data,
            &mut GpuSamplerRequest::Mirostat2 {
                sampler: &mut sampler,
                params,
            },
            None,
            None,
        )
        .unwrap();
        let result = Tensor::from(output).as_slice::<1, u32>().await.unwrap();

        assert_eq!(result.as_slice()[0], GPU_SAMPLE_STATUS_SAMPLED);
        assert_eq!(result.as_slice()[1], expected);
    });
}

#[test]
fn retry_status_does_not_mutate_mirostat_state() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let values = [8.0, 7.0, 6.0, 5.0];
        let ids = [0u32, 1, 2, 3];
        let value_buffer = device.create_buffer_init(
            bytemuck::cast_slice(&values),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let id_buffer = device.create_buffer_init(
            bytemuck::cast_slice(&ids),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let retry_flag = [0u32];
        let retry_buffer = device.create_buffer_init(
            bytemuck::cast_slice(&retry_flag),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let values_data =
            TensorData::new_from_buffer(&device, value_buffer, &[values.len()], DataTypeEnum::F32);
        let ids_data =
            TensorData::new_from_buffer(&device, id_buffer, &[ids.len()], DataTypeEnum::U32);
        let exactness_flag =
            TensorData::new_from_buffer(&device, retry_buffer, &[1], DataTypeEnum::U32);
        let mu = 7.25;
        let params = GpuMirostat2SamplerParams {
            top_k: values.len(),
            temperature: 1.0,
            repetition_penalty: 1.0,
            tau: 5.0,
            eta: 0.1,
            random: 0.5,
        };
        let mut sampler = GpuMirostat2Sampler::new(&device, mu);

        let output = sample_from_sorted_top_k_data_with_encoder(
            &ids_data,
            &values_data,
            &mut GpuSamplerRequest::Mirostat2 {
                sampler: &mut sampler,
                params,
            },
            Some(&exactness_flag),
            None,
        )
        .unwrap();
        let result = Tensor::from(output).as_slice::<1, u32>().await.unwrap();
        let state = Tensor::from(sampler.state.clone())
            .as_slice::<1, f32>()
            .await
            .unwrap();

        assert_eq!(result.as_slice()[0], GPU_SAMPLE_STATUS_RETRY_NEEDED);
        assert_eq!(state.as_slice()[0], mu);
    });
}

#[test]
fn mirostat2_sampler_uses_exact_top_k_when_candidates_cluster() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let mut values = vec![0.0f32; 512];
        for (token_id, value) in values.iter_mut().take(128).enumerate() {
            *value = 10.0 - token_id as f32 * 0.005;
        }
        let buffer = device.create_buffer_init(
            bytemuck::cast_slice(&values),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let data = TensorData::new_from_buffer(&device, buffer, &[values.len()], DataTypeEnum::F32);
        let mu = 7.297_829;
        let params = GpuMirostat2SamplerParams {
            top_k: 128,
            temperature: 1.0,
            repetition_penalty: 1.0,
            tau: 5.0,
            eta: 0.1,
            random: 0.99,
        };
        let expected = cpu_mirostat2_selected_token(&values, mu, params);
        assert!(
            expected >= 64,
            "test setup should select a token missing from a 64-candidate chunk"
        );

        let mut sampler = GpuMirostat2Sampler::new(&device, mu);
        let logits = Tensor::from(data);
        let token = sample_token_to_host(
            &logits.data,
            GpuSamplerRequest::Mirostat2 {
                sampler: &mut sampler,
                params,
            },
            &[],
        )
        .await
        .unwrap();

        assert_eq!(token, Some(expected));
    });
}

#[test]
fn top_k_pairs_merge_path_match_cpu_sorted_order() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let values = (0..4096)
            .map(|index| {
                if index % 997 == 0 {
                    f32::NAN
                } else if index % 991 == 0 {
                    f32::INFINITY
                } else if index % 983 == 0 {
                    f32::NEG_INFINITY
                } else {
                    let coarse = ((index * 37) % 251) as f32;
                    let tied = (index % 17) as f32 * 0.001;
                    coarse - tied
                }
            })
            .collect::<Vec<_>>();
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(16).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(16);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    });
}

#[test]
fn top_k_pairs_large_vocab_merge_path_matches_cpu_sorted_order() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let values = (0..128_256)
            .map(|index| {
                if index % 65_521 == 0 {
                    f32::NAN
                } else if index % 32_749 == 0 {
                    f32::INFINITY
                } else if index % 32_719 == 0 {
                    f32::NEG_INFINITY
                } else {
                    let coarse = ((index * 97) % 4093) as f32;
                    let tied = (index % 31) as f32 * 0.0001;
                    coarse - tied
                }
            })
            .collect::<Vec<_>>();
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(512).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(512);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    });
}

#[test]
fn qmat_logits_sample_token_through_graph() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let hidden = Tensor::new(&device, vec![1.0f32; 32].as_slice());
        let element_count = 8 * 32;
        let block_count = element_count / BlockQ4_0::BLOCK_SIZE;
        let raw_bytes = vec![0u8; block_count * size_of::<BlockQ4_0>()];
        let matrix =
            QMatrix::from_parts(&device, &raw_bytes, Box::new([8, 32]), GgmlType::Q4_0).unwrap();
        let mut sampler = GpuMirostat2Sampler::new(&device, 10.0);
        let params = GpuMirostat2SamplerParams {
            top_k: 4,
            temperature: 1.0,
            repetition_penalty: 1.0,
            tau: 5.0,
            eta: 0.1,
            random: 0.0,
        };

        let logits = hidden.q_mat_mul(&matrix);
        let token = sample_token_to_host(
            &logits.data,
            GpuSamplerRequest::Mirostat2 {
                sampler: &mut sampler,
                params,
            },
            &[],
        )
        .await
        .unwrap();

        assert_eq!(token, Some(7));
    });
}
