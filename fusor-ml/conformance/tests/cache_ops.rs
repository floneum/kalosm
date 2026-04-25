use fusor::{
    Device, Tensor,
    cache::{AttentionMask, KvCache, MaskCache, TensorCache},
};
use fusor_conformance::{available_devices, exact_eq};

fn tensor_data(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i % 11) as f32) - 5.0) * 0.25 + offset)
        .collect()
}

async fn assert_attention_mask_apply_3d_case(batch: usize, seq_len: usize) {
    let scores_data = tensor_data(batch * seq_len * seq_len, 0.5);
    let cpu_mask: AttentionMask<f32> = AttentionMask::causal(&Device::Cpu, seq_len);
    let cpu_scores = Tensor::from_slice(&Device::Cpu, [batch, seq_len, seq_len], &scores_data);
    let expected = cpu_mask.apply(&cpu_scores).to_concrete();

    for device in available_devices().await {
        let mask: AttentionMask<f32> = AttentionMask::causal(&device, seq_len);
        let scores = Tensor::from_slice(&device, [batch, seq_len, seq_len], &scores_data);
        let actual = mask.apply(&scores).to_concrete();
        exact_eq(&actual, &expected).await.unwrap();
    }
}

async fn assert_attention_mask_apply_4d_case(batch: usize, heads: usize, seq_len: usize) {
    let scores_data = tensor_data(batch * heads * seq_len * seq_len, -0.25);
    let cpu_mask: AttentionMask<f32> = AttentionMask::causal(&Device::Cpu, seq_len);
    let cpu_scores =
        Tensor::from_slice(&Device::Cpu, [batch, heads, seq_len, seq_len], &scores_data);
    let expected = cpu_mask.apply(&cpu_scores).to_concrete();

    for device in available_devices().await {
        let mask: AttentionMask<f32> = AttentionMask::causal(&device, seq_len);
        let scores = Tensor::from_slice(&device, [batch, heads, seq_len, seq_len], &scores_data);
        let actual = mask.apply(&scores).to_concrete();
        exact_eq(&actual, &expected).await.unwrap();
    }
}

#[tokio::test]
async fn attention_mask_causal_matches_expected_on_varied_sizes() {
    for seq_len in [1, 2, 4, 7] {
        let expected: AttentionMask<f32> = AttentionMask::causal(&Device::Cpu, seq_len);
        for device in available_devices().await {
            let mask: AttentionMask<f32> = AttentionMask::causal(&device, seq_len);
            exact_eq(mask.mask(), expected.mask()).await.unwrap();
        }
    }
}

#[tokio::test]
async fn attention_mask_apply_broadcasts_to_varied_3d_and_4d_shapes() {
    for (batch, seq_len) in [(1, 2), (2, 3), (3, 4)] {
        assert_attention_mask_apply_3d_case(batch, seq_len).await;
    }
    for (batch, heads, seq_len) in [(1, 1, 2), (2, 3, 3), (2, 2, 5)] {
        assert_attention_mask_apply_4d_case(batch, heads, seq_len).await;
    }
}

#[tokio::test]
async fn tensor_cache_append_and_reset_work_across_varied_cases() {
    for device in available_devices().await {
        for &(max_sequence_len, batch, features, chunk_lens) in &[
            (5usize, 1usize, 2usize, &[1usize, 2, 1][..]),
            (4usize, 2usize, 3usize, &[2usize, 2, 1][..]),
        ] {
            let mut expected: TensorCache<3, f32> = TensorCache::new(1, max_sequence_len);
            let mut actual: TensorCache<3, f32> = TensorCache::new(1, max_sequence_len);

            for (step, &chunk_len) in chunk_lens.iter().enumerate() {
                let data = tensor_data(batch * chunk_len * features, step as f32 + 0.25);
                let cpu_tensor =
                    Tensor::from_slice(&Device::Cpu, [batch, chunk_len, features], &data);
                let device_tensor =
                    Tensor::from_slice(&device, [batch, chunk_len, features], &data);

                let expected_tensor = expected.append(&Device::Cpu, &cpu_tensor);
                let actual_tensor = actual.append(&device, &device_tensor);
                exact_eq(&actual_tensor, &expected_tensor).await.unwrap();
                assert_eq!(actual.current_seq_len(), expected.current_seq_len());
            }

            actual.reset();
            assert_eq!(actual.current_seq_len(), 0);
            assert!(actual.current_data().is_none());
        }

        for &(max_sequence_len, batch, channels, chunk_lens) in &[
            (6usize, 1usize, 2usize, &[1usize, 3, 2][..]),
            (5usize, 2usize, 3usize, &[2usize, 1, 3][..]),
        ] {
            let mut expected: TensorCache<3, f32> = TensorCache::new(2, max_sequence_len);
            let mut actual: TensorCache<3, f32> = TensorCache::new(2, max_sequence_len);

            for (step, &chunk_len) in chunk_lens.iter().enumerate() {
                let data = tensor_data(batch * channels * chunk_len, step as f32 + 1.5);
                let cpu_tensor =
                    Tensor::from_slice(&Device::Cpu, [batch, channels, chunk_len], &data);
                let device_tensor =
                    Tensor::from_slice(&device, [batch, channels, chunk_len], &data);

                let expected_tensor = expected.append(&Device::Cpu, &cpu_tensor);
                let actual_tensor = actual.append(&device, &device_tensor);
                exact_eq(&actual_tensor, &expected_tensor).await.unwrap();
                assert_eq!(actual.current_seq_len(), expected.current_seq_len());
            }
        }
    }
}

#[tokio::test]
async fn kv_cache_append_and_reset_work_across_varied_cases() {
    for device in available_devices().await {
        for &(max_sequence_len, batch, heads, dim, chunk_lens) in &[
            (4usize, 1usize, 1usize, 2usize, &[1usize, 2, 1][..]),
            (5usize, 2usize, 3usize, 4usize, &[2usize, 1, 2][..]),
        ] {
            let mut expected: KvCache<f32> = KvCache::new(1, max_sequence_len);
            let mut actual: KvCache<f32> = KvCache::new(1, max_sequence_len);

            for (step, &chunk_len) in chunk_lens.iter().enumerate() {
                let key_data = tensor_data(batch * chunk_len * heads * dim, step as f32 + 0.75);
                let value_data = tensor_data(batch * chunk_len * heads * dim, step as f32 + 2.25);
                let cpu_key =
                    Tensor::from_slice(&Device::Cpu, [batch, chunk_len, heads, dim], &key_data);
                let cpu_value =
                    Tensor::from_slice(&Device::Cpu, [batch, chunk_len, heads, dim], &value_data);
                let device_key =
                    Tensor::from_slice(&device, [batch, chunk_len, heads, dim], &key_data);
                let device_value =
                    Tensor::from_slice(&device, [batch, chunk_len, heads, dim], &value_data);

                let (expected_keys, expected_values) =
                    expected.append(&Device::Cpu, &cpu_key, &cpu_value);
                let (actual_keys, actual_values) =
                    actual.append(&device, &device_key, &device_value);
                exact_eq(&actual_keys, &expected_keys).await.unwrap();
                exact_eq(&actual_values, &expected_values).await.unwrap();
                assert_eq!(actual.current_seq_len(), expected.current_seq_len());
            }

            actual.reset();
            assert!(actual.k().is_none());
            assert!(actual.v().is_none());
            assert_eq!(actual.current_seq_len(), 0);
        }
    }
}

#[tokio::test]
async fn mask_cache_supports_varied_offsets_and_sliding_windows() {
    for &(seq_len, offset, sliding_window) in &[
        (1usize, 0usize, None),
        (3usize, 2usize, None),
        (4usize, 0usize, Some(2usize)),
        (5usize, 3usize, Some(4usize)),
    ] {
        let cpu_cache: MaskCache<f32> = MaskCache::default();
        let expected = cpu_cache.get_mask(seq_len, offset, sliding_window, &Device::Cpu);

        for device in available_devices().await {
            let cache: MaskCache<f32> = MaskCache::default();
            let actual = cache.get_mask(seq_len, offset, sliding_window, &device);
            exact_eq(actual.mask(), expected.mask()).await.unwrap();
        }
    }
}
