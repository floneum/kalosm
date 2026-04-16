use fusor::{
    Device, Tensor,
    layers::{Conv1d, Conv1dConfig, Embedding, LayerNorm, RmsNorm},
};
use fusor_conformance::{approx_eq, available_devices, exact_eq};

#[tokio::test]
async fn conv1d_matches_expected_values_on_available_devices() {
    let expected = Tensor::from_slice(&Device::Cpu, [1, 1, 3], &[2.2f32, 3.2, 4.2]);

    for device in available_devices().await {
        let weight = Tensor::from_slice(&device, [1, 1, 3], &[0.2f32, 0.5, 0.3]);
        let bias = Tensor::from_slice(&device, [1], &[0.1f32]);
        let input = Tensor::from_slice(&device, [1, 1, 5], &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let conv = Conv1d::new(weight, Some(bias), Conv1dConfig::default());
        let actual = conv.forward(&input);
        approx_eq(&actual, &expected, 1e-5).await.unwrap();
    }
}

#[tokio::test]
async fn conv1d_with_padding_matches_expected_values_on_available_devices() {
    let expected = Tensor::from_slice(&Device::Cpu, [1, 1, 3], &[3.0f32, 6.0, 5.0]);

    for device in available_devices().await {
        let weight = Tensor::from_slice(&device, [1, 1, 3], &[1.0f32, 1.0, 1.0]);
        let input = Tensor::from_slice(&device, [1, 1, 3], &[1.0f32, 2.0, 3.0]);
        let conv = Conv1d::new(
            weight,
            None,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let actual = conv.forward(&input);
        approx_eq(&actual, &expected, 1e-5).await.unwrap();
    }
}

#[tokio::test]
async fn conv1d_properties_match_configuration() {
    let weight = Tensor::from_slice(&Device::Cpu, [2, 3, 1], &[0.0f32; 6]);
    let conv = Conv1d::new(
        weight,
        None,
        Conv1dConfig {
            padding: 2,
            stride: 3,
            ..Default::default()
        },
    );

    assert_eq!(conv.in_channels(), 3);
    assert_eq!(conv.out_channels(), 2);
    assert_eq!(conv.kernel_size(), 1);
    assert_eq!(conv.config().padding, 2);
    assert_eq!(conv.config().stride, 3);
}

#[tokio::test]
async fn embedding_lookup_and_properties_match_expected_on_available_devices() {
    for device in available_devices().await {
        let embeddings = Tensor::from_slice(&device, [3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let layer = Embedding::new_from_tensor(embeddings);

        let indices_1d = Tensor::from_slice(&device, [3], &[0u32, 2, 1]);
        let result_1d: Tensor<2, f32> = layer.forward(&indices_1d);
        let expected_1d =
            Tensor::from_slice(&Device::Cpu, [3, 2], &[1.0f32, 2.0, 5.0, 6.0, 3.0, 4.0]);
        exact_eq(&result_1d, &expected_1d).await.unwrap();

        let indices_2d = Tensor::from_slice(&device, [2, 2], &[0u32, 1, 2, 0]);
        let result_2d: Tensor<3, f32> = layer.forward(&indices_2d);
        let expected_2d = Tensor::from_slice(
            &Device::Cpu,
            [2, 2, 2],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0],
        );
        exact_eq(&result_2d, &expected_2d).await.unwrap();

        assert_eq!(layer.num_embeddings(), 3);
        assert_eq!(layer.embedding_dim(), 2);
    }
}

#[tokio::test]
async fn layer_norm_matches_expected_values_on_available_devices() {
    let expected_val = (3.0f32 / 2.0).sqrt();
    let expected_2d = Tensor::from_slice(
        &Device::Cpu,
        [2, 3],
        &[
            -expected_val,
            0.0,
            expected_val,
            -expected_val,
            0.0,
            expected_val,
        ],
    );
    let expected_3d = Tensor::from_slice(&Device::Cpu, [1, 2, 2], &[-1.0f32, 1.0, -1.0, 1.0]);

    for device in available_devices().await {
        let weight_2d = Tensor::from_slice(&device, [3], &[1.0f32, 1.0, 1.0]);
        let bias_2d = Tensor::from_slice(&device, [3], &[0.0f32, 0.0, 0.0]);
        let layer_norm_2d = LayerNorm::new(weight_2d, Some(bias_2d), 1e-5);
        let input_2d = Tensor::from_slice(&device, [2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let actual_2d = layer_norm_2d.forward_2d(&input_2d);
        approx_eq(&actual_2d, &expected_2d, 1e-4).await.unwrap();

        let weight_3d = Tensor::from_slice(&device, [2], &[1.0f32, 1.0]);
        let layer_norm_3d = LayerNorm::new(weight_3d, None, 1e-5);
        let input_3d = Tensor::from_slice(&device, [1, 2, 2], &[1.0f32, 3.0, 2.0, 4.0]);
        let actual_3d = layer_norm_3d.forward(&input_3d);
        approx_eq(&actual_3d, &expected_3d, 1e-4).await.unwrap();
    }
}

#[tokio::test]
async fn layer_norm_fused_cpu_matches_reference() {
    let weight = Tensor::from_slice(&Device::Cpu, [2], &[1.5f32, 0.5]);
    let bias = Tensor::from_slice(&Device::Cpu, [2], &[0.25f32, -0.75]);
    let layer_norm = LayerNorm::new(weight, Some(bias), 1e-5);
    let input = Tensor::from_slice(&Device::Cpu, [1, 2, 2], &[1.0f32, 3.0, 2.0, 4.0]);

    let fused = layer_norm.forward_fused(&input);
    let reference = layer_norm.forward(&input);
    approx_eq(&fused, &reference, 1e-4).await.unwrap();
}

#[tokio::test]
async fn rms_norm_matches_expected_values_on_available_devices() {
    let rms_2d = ((1.0f32 + 4.0 + 9.0) / 3.0).sqrt();
    let expected_2d = Tensor::from_slice(
        &Device::Cpu,
        [2, 3],
        &[
            1.0 / rms_2d,
            2.0 / rms_2d,
            3.0 / rms_2d,
            4.0 / ((16.0f32 + 25.0 + 36.0) / 3.0).sqrt(),
            5.0 / ((16.0f32 + 25.0 + 36.0) / 3.0).sqrt(),
            6.0 / ((16.0f32 + 25.0 + 36.0) / 3.0).sqrt(),
        ],
    );
    let rms_3d_a = ((9.0f32 + 16.0) / 2.0).sqrt();
    let rms_3d_b = ((36.0f32 + 64.0) / 2.0).sqrt();
    let expected_3d = Tensor::from_slice(
        &Device::Cpu,
        [1, 2, 2],
        &[
            3.0 / rms_3d_a * 2.0,
            4.0 / rms_3d_a * 2.0,
            6.0 / rms_3d_b * 2.0,
            8.0 / rms_3d_b * 2.0,
        ],
    );

    for device in available_devices().await {
        let weight_2d = Tensor::from_slice(&device, [3], &[1.0f32, 1.0, 1.0]);
        let rms_norm_2d = RmsNorm::new(weight_2d, None, 1e-5);
        let input_2d = Tensor::from_slice(&device, [2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let actual_2d = rms_norm_2d.forward_2d(&input_2d);
        approx_eq(&actual_2d, &expected_2d, 1e-4).await.unwrap();

        let weight_3d = Tensor::from_slice(&device, [2], &[2.0f32, 2.0]);
        let rms_norm_3d = RmsNorm::new(weight_3d, None, 1e-5);
        let input_3d = Tensor::from_slice(&device, [1, 2, 2], &[3.0f32, 4.0, 6.0, 8.0]);
        let actual_3d = rms_norm_3d.forward(&input_3d);
        approx_eq(&actual_3d, &expected_3d, 1e-4).await.unwrap();
    }
}
