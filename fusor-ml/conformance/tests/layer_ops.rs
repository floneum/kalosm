use fusor::{
    Device, Tensor,
    layers::{Conv1d, Conv1dConfig, Embedding, LayerNorm, RmsNorm},
};
use fusor_conformance::{approx_eq, available_devices, exact_eq};

#[derive(Clone, Copy)]
struct ConvCase {
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    length: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    with_bias: bool,
}

fn layer_data(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i % 19) as f32) - 9.0) * 0.15 + offset)
        .collect()
}

fn index_data(len: usize, modulo: usize) -> Vec<u32> {
    (0..len).map(|i| ((i * 3 + 1) % modulo) as u32).collect()
}

async fn assert_conv1d_case(case: ConvCase) {
    let input_data = layer_data(case.batch * case.in_channels * case.length, 0.25);
    let weight_data = layer_data(
        case.out_channels * case.in_channels * case.kernel_size,
        -0.35,
    );
    let bias_data = case.with_bias.then(|| layer_data(case.out_channels, 0.1));
    let config = Conv1dConfig {
        padding: case.padding,
        stride: case.stride,
        ..Default::default()
    };

    let input_cpu = Tensor::from_slice(
        &Device::Cpu,
        [case.batch, case.in_channels, case.length],
        &input_data,
    );
    let weight_cpu = Tensor::from_slice(
        &Device::Cpu,
        [case.out_channels, case.in_channels, case.kernel_size],
        &weight_data,
    );
    let bias_cpu = bias_data
        .as_ref()
        .map(|data| Tensor::from_slice(&Device::Cpu, [case.out_channels], data));
    let expected = Conv1d::new(weight_cpu, bias_cpu, config)
        .forward(&input_cpu)
        .to_concrete();

    for device in available_devices().await {
        let input = Tensor::from_slice(
            &device,
            [case.batch, case.in_channels, case.length],
            &input_data,
        );
        let weight = Tensor::from_slice(
            &device,
            [case.out_channels, case.in_channels, case.kernel_size],
            &weight_data,
        );
        let bias = bias_data
            .as_ref()
            .map(|data| Tensor::from_slice(&device, [case.out_channels], data));
        let actual = Conv1d::new(weight, bias, config)
            .forward(&input)
            .to_concrete();
        approx_eq(&actual, &expected, 1e-5).await.unwrap();
    }
}

async fn assert_embedding_1d_case(num_embeddings: usize, embedding_dim: usize, len: usize) {
    let embedding_data = layer_data(num_embeddings * embedding_dim, -0.2);
    let indices_data = index_data(len, num_embeddings);
    let cpu_embeddings = Tensor::from_slice(
        &Device::Cpu,
        [num_embeddings, embedding_dim],
        &embedding_data,
    );
    let cpu_indices = Tensor::from_slice(&Device::Cpu, [len], &indices_data);
    let cpu_layer = Embedding::new_from_tensor(cpu_embeddings);
    let expected: Tensor<2, f32> = cpu_layer.forward(&cpu_indices);

    for device in available_devices().await {
        let embeddings =
            Tensor::from_slice(&device, [num_embeddings, embedding_dim], &embedding_data);
        let indices = Tensor::from_slice(&device, [len], &indices_data);
        let layer = Embedding::new_from_tensor(embeddings);
        let actual: Tensor<2, f32> = layer.forward(&indices);
        exact_eq(&actual, &expected).await.unwrap();
        assert_eq!(layer.num_embeddings(), num_embeddings);
        assert_eq!(layer.embedding_dim(), embedding_dim);
    }
}

async fn assert_embedding_2d_case(
    num_embeddings: usize,
    embedding_dim: usize,
    batch: usize,
    seq_len: usize,
) {
    let embedding_data = layer_data(num_embeddings * embedding_dim, 0.4);
    let indices_data = index_data(batch * seq_len, num_embeddings);
    let cpu_embeddings = Tensor::from_slice(
        &Device::Cpu,
        [num_embeddings, embedding_dim],
        &embedding_data,
    );
    let cpu_indices = Tensor::from_slice(&Device::Cpu, [batch, seq_len], &indices_data);
    let cpu_layer = Embedding::new_from_tensor(cpu_embeddings);
    let expected: Tensor<3, f32> = cpu_layer.forward(&cpu_indices);

    for device in available_devices().await {
        let embeddings =
            Tensor::from_slice(&device, [num_embeddings, embedding_dim], &embedding_data);
        let indices = Tensor::from_slice(&device, [batch, seq_len], &indices_data);
        let layer = Embedding::new_from_tensor(embeddings);
        let actual: Tensor<3, f32> = layer.forward(&indices);
        exact_eq(&actual, &expected).await.unwrap();
    }
}

async fn assert_embedding_3d_case(
    num_embeddings: usize,
    embedding_dim: usize,
    batch: usize,
    heads: usize,
    seq_len: usize,
) {
    let embedding_data = layer_data(num_embeddings * embedding_dim, -0.6);
    let indices_data = index_data(batch * heads * seq_len, num_embeddings);
    let cpu_embeddings = Tensor::from_slice(
        &Device::Cpu,
        [num_embeddings, embedding_dim],
        &embedding_data,
    );
    let cpu_indices = Tensor::from_slice(&Device::Cpu, [batch, heads, seq_len], &indices_data);
    let cpu_layer = Embedding::new_from_tensor(cpu_embeddings);
    let expected: Tensor<4, f32> = cpu_layer.forward(&cpu_indices);

    for device in available_devices().await {
        let embeddings =
            Tensor::from_slice(&device, [num_embeddings, embedding_dim], &embedding_data);
        let indices = Tensor::from_slice(&device, [batch, heads, seq_len], &indices_data);
        let layer = Embedding::new_from_tensor(embeddings);
        let actual: Tensor<4, f32> = layer.forward(&indices);
        exact_eq(&actual, &expected).await.unwrap();
    }
}

async fn assert_layer_norm_2d_case(batch: usize, features: usize, with_bias: bool) {
    let input_data = layer_data(batch * features, 0.15);
    let weight_data = layer_data(features, 1.0);
    let bias_data = with_bias.then(|| layer_data(features, -0.2));
    let cpu_weight = Tensor::from_slice(&Device::Cpu, [features], &weight_data);
    let cpu_bias = bias_data
        .as_ref()
        .map(|data| Tensor::from_slice(&Device::Cpu, [features], data));
    let layer_norm_cpu = LayerNorm::new(cpu_weight, cpu_bias, 1e-5);
    let input_cpu = Tensor::from_slice(&Device::Cpu, [batch, features], &input_data);
    let expected = layer_norm_cpu.forward_2d(&input_cpu).to_concrete();

    for device in available_devices().await {
        let weight = Tensor::from_slice(&device, [features], &weight_data);
        let bias = bias_data
            .as_ref()
            .map(|data| Tensor::from_slice(&device, [features], data));
        let layer_norm = LayerNorm::new(weight, bias, 1e-5);
        let input = Tensor::from_slice(&device, [batch, features], &input_data);
        let actual = layer_norm.forward_2d(&input).to_concrete();
        approx_eq(&actual, &expected, 1e-4).await.unwrap();
    }
}

async fn assert_layer_norm_3d_case(batch: usize, seq_len: usize, features: usize, with_bias: bool) {
    let input_data = layer_data(batch * seq_len * features, -0.4);
    let weight_data = layer_data(features, 0.8);
    let bias_data = with_bias.then(|| layer_data(features, 0.3));
    let cpu_weight = Tensor::from_slice(&Device::Cpu, [features], &weight_data);
    let cpu_bias = bias_data
        .as_ref()
        .map(|data| Tensor::from_slice(&Device::Cpu, [features], data));
    let layer_norm_cpu = LayerNorm::new(cpu_weight, cpu_bias, 1e-5);
    let input_cpu = Tensor::from_slice(&Device::Cpu, [batch, seq_len, features], &input_data);
    let expected = layer_norm_cpu.forward(&input_cpu).to_concrete();

    for device in available_devices().await {
        let weight = Tensor::from_slice(&device, [features], &weight_data);
        let bias = bias_data
            .as_ref()
            .map(|data| Tensor::from_slice(&device, [features], data));
        let layer_norm = LayerNorm::new(weight, bias, 1e-5);
        let input = Tensor::from_slice(&device, [batch, seq_len, features], &input_data);
        let actual = layer_norm.forward(&input).to_concrete();
        approx_eq(&actual, &expected, 1e-4).await.unwrap();
    }
}

async fn assert_rms_norm_2d_case(batch: usize, features: usize, with_bias: bool) {
    let input_data = layer_data(batch * features, 0.5);
    let weight_data = layer_data(features, 1.2);
    let bias_data = with_bias.then(|| layer_data(features, -0.3));
    let cpu_weight = Tensor::from_slice(&Device::Cpu, [features], &weight_data);
    let cpu_bias = bias_data
        .as_ref()
        .map(|data| Tensor::from_slice(&Device::Cpu, [features], data));
    let rms_norm_cpu = RmsNorm::new(cpu_weight, cpu_bias, 1e-5);
    let input_cpu = Tensor::from_slice(&Device::Cpu, [batch, features], &input_data);
    let expected = rms_norm_cpu.forward_2d(&input_cpu).to_concrete();

    for device in available_devices().await {
        let weight = Tensor::from_slice(&device, [features], &weight_data);
        let bias = bias_data
            .as_ref()
            .map(|data| Tensor::from_slice(&device, [features], data));
        let rms_norm = RmsNorm::new(weight, bias, 1e-5);
        let input = Tensor::from_slice(&device, [batch, features], &input_data);
        let actual = rms_norm.forward_2d(&input).to_concrete();
        approx_eq(&actual, &expected, 1e-4).await.unwrap();
    }
}

async fn assert_rms_norm_3d_case(batch: usize, seq_len: usize, features: usize, with_bias: bool) {
    let input_data = layer_data(batch * seq_len * features, -0.55);
    let weight_data = layer_data(features, 0.95);
    let bias_data = with_bias.then(|| layer_data(features, 0.2));
    let cpu_weight = Tensor::from_slice(&Device::Cpu, [features], &weight_data);
    let cpu_bias = bias_data
        .as_ref()
        .map(|data| Tensor::from_slice(&Device::Cpu, [features], data));
    let rms_norm_cpu = RmsNorm::new(cpu_weight, cpu_bias, 1e-5);
    let input_cpu = Tensor::from_slice(&Device::Cpu, [batch, seq_len, features], &input_data);
    let expected = rms_norm_cpu.forward(&input_cpu).to_concrete();

    for device in available_devices().await {
        let weight = Tensor::from_slice(&device, [features], &weight_data);
        let bias = bias_data
            .as_ref()
            .map(|data| Tensor::from_slice(&device, [features], data));
        let rms_norm = RmsNorm::new(weight, bias, 1e-5);
        let input = Tensor::from_slice(&device, [batch, seq_len, features], &input_data);
        let actual = rms_norm.forward(&input).to_concrete();
        approx_eq(&actual, &expected, 1e-4).await.unwrap();
    }
}

async fn assert_rms_norm_4d_case(
    batch: usize,
    heads: usize,
    seq_len: usize,
    features: usize,
    with_bias: bool,
) {
    let input_data = layer_data(batch * heads * seq_len * features, 0.7);
    let weight_data = layer_data(features, 1.1);
    let bias_data = with_bias.then(|| layer_data(features, -0.1));
    let cpu_weight = Tensor::from_slice(&Device::Cpu, [features], &weight_data);
    let cpu_bias = bias_data
        .as_ref()
        .map(|data| Tensor::from_slice(&Device::Cpu, [features], data));
    let rms_norm_cpu = RmsNorm::new(cpu_weight, cpu_bias, 1e-5);
    let input_cpu =
        Tensor::from_slice(&Device::Cpu, [batch, heads, seq_len, features], &input_data);
    let expected = rms_norm_cpu.forward_4d(&input_cpu).to_concrete();

    for device in available_devices().await {
        let weight = Tensor::from_slice(&device, [features], &weight_data);
        let bias = bias_data
            .as_ref()
            .map(|data| Tensor::from_slice(&device, [features], data));
        let rms_norm = RmsNorm::new(weight, bias, 1e-5);
        let input = Tensor::from_slice(&device, [batch, heads, seq_len, features], &input_data);
        let actual = rms_norm.forward_4d(&input).to_concrete();
        approx_eq(&actual, &expected, 1e-4).await.unwrap();
    }
}

#[tokio::test]
async fn conv1d_matches_cpu_reference_on_varied_shapes() {
    for case in [
        ConvCase {
            batch: 1,
            in_channels: 1,
            out_channels: 1,
            length: 5,
            kernel_size: 3,
            padding: 0,
            stride: 1,
            with_bias: true,
        },
        ConvCase {
            batch: 2,
            in_channels: 2,
            out_channels: 3,
            length: 8,
            kernel_size: 2,
            padding: 1,
            stride: 1,
            with_bias: false,
        },
        ConvCase {
            batch: 1,
            in_channels: 3,
            out_channels: 2,
            length: 10,
            kernel_size: 3,
            padding: 2,
            stride: 2,
            with_bias: true,
        },
    ] {
        assert_conv1d_case(case).await;
    }
}

#[tokio::test]
async fn conv1d_properties_match_configuration() {
    for &(out_channels, in_channels, kernel_size, padding, stride) in
        &[(2usize, 3usize, 1usize, 2usize, 3usize), (4, 2, 5, 1, 2)]
    {
        let weight = Tensor::from_slice(
            &Device::Cpu,
            [out_channels, in_channels, kernel_size],
            &vec![0.0f32; out_channels * in_channels * kernel_size],
        );
        let conv = Conv1d::new(
            weight,
            None,
            Conv1dConfig {
                padding,
                stride,
                ..Default::default()
            },
        );

        assert_eq!(conv.in_channels(), in_channels);
        assert_eq!(conv.out_channels(), out_channels);
        assert_eq!(conv.kernel_size(), kernel_size);
        assert_eq!(conv.config().padding, padding);
        assert_eq!(conv.config().stride, stride);
    }
}

#[tokio::test]
async fn embedding_lookup_matches_cpu_reference_on_varied_shapes() {
    assert_embedding_1d_case(5, 3, 6).await;
    assert_embedding_2d_case(7, 4, 2, 5).await;
    assert_embedding_3d_case(6, 5, 2, 3, 4).await;
}

#[tokio::test]
async fn layer_norm_matches_cpu_reference_on_varied_shapes() {
    for &(batch, features, with_bias) in &[(2usize, 3usize, false), (3, 5, true), (4, 7, true)] {
        assert_layer_norm_2d_case(batch, features, with_bias).await;
    }
    for &(batch, seq_len, features, with_bias) in &[
        (1usize, 2usize, 2usize, false),
        (2, 3, 4, true),
        (3, 2, 6, true),
    ] {
        assert_layer_norm_3d_case(batch, seq_len, features, with_bias).await;
    }
}

#[tokio::test]
async fn layer_norm_fused_cpu_matches_reference_on_varied_shapes() {
    for &(batch, seq_len, features) in &[(1usize, 2usize, 2usize), (2, 3, 4), (1, 4, 7)] {
        let weight_data = layer_data(features, 1.3);
        let bias_data = layer_data(features, -0.45);
        let input_data = layer_data(batch * seq_len * features, 0.2);
        let weight = Tensor::from_slice(&Device::Cpu, [features], &weight_data);
        let bias = Tensor::from_slice(&Device::Cpu, [features], &bias_data);
        let layer_norm = LayerNorm::new(weight, Some(bias), 1e-5);
        let input = Tensor::from_slice(&Device::Cpu, [batch, seq_len, features], &input_data);

        let fused = layer_norm.forward_fused(&input).to_concrete();
        let reference = layer_norm.forward(&input).to_concrete();
        approx_eq(&fused, &reference, 1e-4).await.unwrap();
    }
}

#[tokio::test]
async fn rms_norm_matches_cpu_reference_on_varied_shapes() {
    for &(batch, features, with_bias) in &[(2usize, 3usize, false), (3, 5, true), (4, 6, true)] {
        assert_rms_norm_2d_case(batch, features, with_bias).await;
    }
    for &(batch, seq_len, features, with_bias) in &[
        (1usize, 2usize, 2usize, false),
        (2, 3, 4, true),
        (3, 2, 5, true),
    ] {
        assert_rms_norm_3d_case(batch, seq_len, features, with_bias).await;
    }
    for &(batch, heads, seq_len, features, with_bias) in
        &[(1usize, 2usize, 3usize, 4usize, false), (2, 3, 2, 5, true)]
    {
        assert_rms_norm_4d_case(batch, heads, seq_len, features, with_bias).await;
    }
}
