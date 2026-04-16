use fusor::{Device, MaskKind, Tensor};
use fusor_conformance::{approx_eq, available_devices};

async fn assert_close<const R: usize>(
    actual: &Tensor<R, f32>,
    expected: &Tensor<R, f32>,
    tol: f32,
) {
    approx_eq(actual, expected, tol).await.unwrap();
}

#[tokio::test]
async fn flash_attention_matches_cpu_reference_on_available_devices() {
    let q_data = [1.0f32, 0.0, 0.0, 1.0];
    let k_data = [1.0f32, 0.0, 0.0, 1.0];
    let v_data = [1.0f32, 2.0, 3.0, 4.0];
    let scale = 1.0 / 2.0_f32.sqrt();

    let q_cpu = Tensor::from_slice(&Device::Cpu, [1, 1, 2, 2], &q_data);
    let k_cpu = Tensor::from_slice(&Device::Cpu, [1, 1, 2, 2], &k_data);
    let v_cpu = Tensor::from_slice(&Device::Cpu, [1, 1, 2, 2], &v_data);
    let expected = q_cpu
        .flash_attention(&k_cpu, &v_cpu, scale, None)
        .to_concrete();

    for device in available_devices().await {
        let q = Tensor::from_slice(&device, [1, 1, 2, 2], &q_data);
        let k = Tensor::from_slice(&device, [1, 1, 2, 2], &k_data);
        let v = Tensor::from_slice(&device, [1, 1, 2, 2], &v_data);
        let actual = q.flash_attention(&k, &v, scale, None).to_concrete();
        assert_close(&actual, &expected, 1e-5).await;
    }
}

#[tokio::test]
async fn flash_attention_with_qk_mask_matches_cpu_reference_on_available_devices() {
    let q_data = [1.0f32, 0.0, 0.0, 1.0];
    let k_data = [1.0f32, 0.0, 0.0, 1.0];
    let v_data = [1.0f32, 2.0, 3.0, 4.0];
    let mask_data = [0.0f32, f32::NEG_INFINITY, 0.0, 0.0];
    let scale = 1.0 / 2.0_f32.sqrt();

    let q_cpu = Tensor::from_slice(&Device::Cpu, [1, 1, 2, 2], &q_data);
    let k_cpu = Tensor::from_slice(&Device::Cpu, [1, 1, 2, 2], &k_data);
    let v_cpu = Tensor::from_slice(&Device::Cpu, [1, 1, 2, 2], &v_data);
    let mask_cpu = Tensor::from_slice(&Device::Cpu, [2, 2], &mask_data);
    let expected = q_cpu
        .flash_attention(&k_cpu, &v_cpu, scale, Some((&mask_cpu, MaskKind::QKMask)))
        .to_concrete();

    let result = expected.as_slice().await.unwrap();
    assert!((result[[0, 0, 0, 0]] - v_data[0]).abs() < 1e-2);
    assert!((result[[0, 0, 0, 1]] - v_data[1]).abs() < 1e-2);

    for device in available_devices().await {
        let q = Tensor::from_slice(&device, [1, 1, 2, 2], &q_data);
        let k = Tensor::from_slice(&device, [1, 1, 2, 2], &k_data);
        let v = Tensor::from_slice(&device, [1, 1, 2, 2], &v_data);
        let mask = Tensor::from_slice(&device, [2, 2], &mask_data);
        let actual = q
            .flash_attention(&k, &v, scale, Some((&mask, MaskKind::QKMask)))
            .to_concrete();
        assert_close(&actual, &expected, 1e-5).await;
    }
}

#[tokio::test]
async fn flash_attention_gqa_matches_cpu_reference_on_available_devices() {
    let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let v_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 + 2.0).collect();
    let scale = 1.0 / 2.0_f32.sqrt();

    let q_cpu = Tensor::from_slice(&Device::Cpu, [1, 4, 2, 2], &q_data);
    let k_cpu = Tensor::from_slice(&Device::Cpu, [1, 2, 2, 2], &k_data);
    let v_cpu = Tensor::from_slice(&Device::Cpu, [1, 2, 2, 2], &v_data);
    let expected = q_cpu
        .flash_attention(&k_cpu, &v_cpu, scale, None)
        .to_concrete();

    for device in available_devices().await {
        let q = Tensor::from_slice(&device, [1, 4, 2, 2], &q_data);
        let k = Tensor::from_slice(&device, [1, 2, 2, 2], &k_data);
        let v = Tensor::from_slice(&device, [1, 2, 2, 2], &v_data);
        let actual = q.flash_attention(&k, &v, scale, None).to_concrete();
        assert_close(&actual, &expected, 1e-5).await;
    }
}

#[tokio::test]
async fn flash_attention_with_batch_key_mask_matches_cpu_reference_on_available_devices() {
    let q_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let v_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 2.0).collect();
    let mask_data = [0.0f32, 0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY];
    let scale = 1.0 / 2.0_f32.sqrt();

    let q_cpu = Tensor::from_slice(&Device::Cpu, [2, 1, 3, 2], &q_data);
    let k_cpu = Tensor::from_slice(&Device::Cpu, [2, 1, 3, 2], &k_data);
    let v_cpu = Tensor::from_slice(&Device::Cpu, [2, 1, 3, 2], &v_data);
    let mask_cpu = Tensor::from_slice(&Device::Cpu, [2, 3], &mask_data);
    let expected = q_cpu
        .flash_attention(
            &k_cpu,
            &v_cpu,
            scale,
            Some((&mask_cpu, MaskKind::BatchKeyMask)),
        )
        .to_concrete();

    for device in available_devices().await {
        let q = Tensor::from_slice(&device, [2, 1, 3, 2], &q_data);
        let k = Tensor::from_slice(&device, [2, 1, 3, 2], &k_data);
        let v = Tensor::from_slice(&device, [2, 1, 3, 2], &v_data);
        let mask = Tensor::from_slice(&device, [2, 3], &mask_data);
        let actual = q
            .flash_attention(&k, &v, scale, Some((&mask, MaskKind::BatchKeyMask)))
            .to_concrete();
        assert_close(&actual, &expected, 1e-5).await;
    }
}
