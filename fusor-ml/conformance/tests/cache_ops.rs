use fusor::{
    Device, Tensor,
    cache::{AttentionMask, KvCache, MaskCache, TensorCache},
};
use fusor_conformance::{available_devices, exact_eq};

#[tokio::test]
async fn attention_mask_causal_matches_expected_on_available_devices() {
    let expected = Tensor::from_slice(
        &Device::Cpu,
        [3, 3],
        &[
            0.0f32,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
        ],
    );

    for device in available_devices().await {
        let mask: AttentionMask<f32> = AttentionMask::causal(&device, 3);
        exact_eq(mask.mask(), &expected).await.unwrap();
    }
}

#[tokio::test]
async fn attention_mask_apply_broadcasts_to_3d_and_4d() {
    let expected_4d = Tensor::from_slice(
        &Device::Cpu,
        [1, 1, 2, 2],
        &[1.0f32, f32::NEG_INFINITY, 3.0, 4.0],
    );
    let expected_3d = Tensor::from_slice(
        &Device::Cpu,
        [1, 2, 2],
        &[1.0f32, f32::NEG_INFINITY, 3.0, 4.0],
    );

    for device in available_devices().await {
        let mask: AttentionMask<f32> = AttentionMask::causal(&device, 2);
        let scores_4d = Tensor::from_slice(&device, [1, 1, 2, 2], &[1.0f32, 2.0, 3.0, 4.0]);
        let scores_3d = Tensor::from_slice(&device, [1, 2, 2], &[1.0f32, 2.0, 3.0, 4.0]);

        let masked_4d = mask.apply(&scores_4d).to_concrete();
        let masked_3d = mask.apply(&scores_3d).to_concrete();

        exact_eq(&masked_4d, &expected_4d).await.unwrap();
        exact_eq(&masked_3d, &expected_3d).await.unwrap();
    }
}

#[tokio::test]
async fn tensor_cache_append_and_reset_work_on_available_devices() {
    let expected_first = Tensor::from_slice(&Device::Cpu, [1, 1, 2], &[1.0f32, 2.0]);
    let expected_second = Tensor::from_slice(&Device::Cpu, [1, 2, 2], &[1.0f32, 2.0, 3.0, 4.0]);

    for device in available_devices().await {
        let mut cache: TensorCache<3, f32> = TensorCache::new(1, 3);

        let tensor1 = Tensor::from_slice(&device, [1, 1, 2], &[1.0f32, 2.0]);
        let tensor2 = Tensor::from_slice(&device, [1, 1, 2], &[3.0f32, 4.0]);

        let first = cache.append(&device, &tensor1);
        exact_eq(&first, &expected_first).await.unwrap();
        assert_eq!(cache.current_seq_len(), 1);

        let second = cache.append(&device, &tensor2);
        exact_eq(&second, &expected_second).await.unwrap();
        assert_eq!(cache.current_seq_len(), 2);

        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);
        assert!(cache.current_data().is_none());
    }
}

#[tokio::test]
async fn kv_cache_append_and_reset_work_on_available_devices() {
    let expected_keys = Tensor::from_slice(&Device::Cpu, [1, 2, 1, 2], &[1.0f32, 2.0, 5.0, 6.0]);
    let expected_values = Tensor::from_slice(&Device::Cpu, [1, 2, 1, 2], &[3.0f32, 4.0, 7.0, 8.0]);

    for device in available_devices().await {
        let mut cache: KvCache<f32> = KvCache::new(1, 3);
        let key1 = Tensor::from_slice(&device, [1, 1, 1, 2], &[1.0f32, 2.0]);
        let value1 = Tensor::from_slice(&device, [1, 1, 1, 2], &[3.0f32, 4.0]);
        let key2 = Tensor::from_slice(&device, [1, 1, 1, 2], &[5.0f32, 6.0]);
        let value2 = Tensor::from_slice(&device, [1, 1, 1, 2], &[7.0f32, 8.0]);

        let (first_k, first_v) = cache.append(&device, &key1, &value1);
        exact_eq(
            &first_k,
            &Tensor::from_slice(&Device::Cpu, [1, 1, 1, 2], &[1.0f32, 2.0]),
        )
        .await
        .unwrap();
        exact_eq(
            &first_v,
            &Tensor::from_slice(&Device::Cpu, [1, 1, 1, 2], &[3.0f32, 4.0]),
        )
        .await
        .unwrap();

        let (keys, values) = cache.append(&device, &key2, &value2);
        exact_eq(&keys, &expected_keys).await.unwrap();
        exact_eq(&values, &expected_values).await.unwrap();
        assert_eq!(cache.current_seq_len(), 2);

        cache.reset();
        assert!(cache.k().is_none());
        assert!(cache.v().is_none());
        assert_eq!(cache.current_seq_len(), 0);
    }
}

#[tokio::test]
async fn mask_cache_supports_offset_and_sliding_window_on_available_devices() {
    let expected_sliding = Tensor::from_slice(
        &Device::Cpu,
        [4, 4],
        &[
            0.0f32,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
        ],
    );

    for device in available_devices().await {
        let cache: MaskCache<f32> = MaskCache::default();

        let mask = cache.get_mask(3, 0, None, &device);
        assert_eq!(mask.mask().shape(), [3, 3]);

        let padded = cache.get_mask(2, 3, None, &device);
        assert_eq!(padded.mask().shape(), [2, 5]);

        let sliding = cache.get_mask(4, 0, Some(2), &device);
        exact_eq(sliding.mask(), &expected_sliding).await.unwrap();
    }
}
