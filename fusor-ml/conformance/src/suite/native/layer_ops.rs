//! Layer op conformance cases.

use fusor::{
    Device, Tensor,
    layers::{ConvNd, ConvNdConfig, Embedding, LayerNorm, RmsNorm},
};
use fusor_conformance::{
    AssertionCase, AssertionCases, approx_compare, exact_compare, exact_value_compare,
};

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

fn assert_conv1d_case(case: ConvCase) -> AssertionCase {
    let input_data = layer_data(case.batch * case.in_channels * case.length, 0.25);
    let weight_data = layer_data(
        case.out_channels * case.in_channels * case.kernel_size,
        -0.35,
    );
    let bias_data = case.with_bias.then(|| layer_data(case.out_channels, 0.1));
    let config = ConvNdConfig {
        padding: [case.padding],
        stride: [case.stride],
        groups: 1,
    };

    fusor_conformance::assert(move |device: Device| {
        let input_data = input_data.clone();
        let weight_data = weight_data.clone();
        let bias_data = bias_data.clone();
        async move {
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
            ConvNd::new(weight, bias, config)
                .forward(&input)
                .to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<3, f32>(1e-5))
    .runs(1)
    .into_case(format!(
        "layer_ops::conv1d_matches_cpu_reference_on_varied_shapes::b{}_ic{}_oc{}_len{}_k{}_pad{}_stride{}_bias{}",
        case.batch,
        case.in_channels,
        case.out_channels,
        case.length,
        case.kernel_size,
        case.padding,
        case.stride,
        case.with_bias
    ))
}

fn assert_embedding_1d_case(
    num_embeddings: usize,
    embedding_dim: usize,
    len: usize,
) -> AssertionCases {
    let embedding_data = layer_data(num_embeddings * embedding_dim, -0.2);
    let indices_data = index_data(len, num_embeddings);
    let mut assertions = AssertionCases::new();

    assertions.push(
        fusor_conformance::assert({
            let embedding_data = embedding_data.clone();
            let indices_data = indices_data.clone();
            move |device: Device| {
                let embedding_data = embedding_data.clone();
                let indices_data = indices_data.clone();
                async move {
                    let embeddings = Tensor::from_slice(
                        &device,
                        [num_embeddings, embedding_dim],
                        &embedding_data,
                    );
                    let indices = Tensor::from_slice(&device, [len], &indices_data);
                    let layer = Embedding::new_from_tensor(embeddings);
                    let actual: Tensor<2, f32> = layer.forward(&indices);
                    actual
                }
            }
        })
        .arg(|device: &Device| device.clone())
        .compare_with(exact_compare::<2, f32>())
        .runs(1)
        .into_case(format!(
            "layer_ops::embedding_lookup_matches_cpu_reference_on_varied_shapes::1d_{num_embeddings}x{embedding_dim}_len{len}"
        )),
    );

    assertions.push(
        fusor_conformance::assert(move |device: Device| {
            let embedding_data = embedding_data.clone();
            async move {
                let embeddings =
                    Tensor::from_slice(&device, [num_embeddings, embedding_dim], &embedding_data);
                let layer = Embedding::new_from_tensor(embeddings);
                (layer.num_embeddings(), layer.embedding_dim())
            }
        })
        .arg(|device: &Device| device.clone())
        .equal_to(move |_device: Device| async move { (num_embeddings, embedding_dim) })
        .compare_with(exact_value_compare())
        .runs(1)
        .into_case(format!(
            "layer_ops::embedding_lookup_matches_cpu_reference_on_varied_shapes::properties_{num_embeddings}x{embedding_dim}"
        )),
    );

    assertions
}

fn assert_embedding_2d_case(
    num_embeddings: usize,
    embedding_dim: usize,
    batch: usize,
    seq_len: usize,
) -> AssertionCase {
    let embedding_data = layer_data(num_embeddings * embedding_dim, 0.4);
    let indices_data = index_data(batch * seq_len, num_embeddings);
    fusor_conformance::assert(move |device: Device| {
        let embedding_data = embedding_data.clone();
        let indices_data = indices_data.clone();
        async move {
            let embeddings =
                Tensor::from_slice(&device, [num_embeddings, embedding_dim], &embedding_data);
            let indices = Tensor::from_slice(&device, [batch, seq_len], &indices_data);
            let layer = Embedding::new_from_tensor(embeddings);
            let actual: Tensor<3, f32> = layer.forward(&indices);
            actual
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(exact_compare::<3, f32>())
    .runs(1)
    .into_case(format!(
        "layer_ops::embedding_lookup_matches_cpu_reference_on_varied_shapes::2d_{num_embeddings}x{embedding_dim}_{batch}x{seq_len}"
    ))
}

fn assert_embedding_3d_case(
    num_embeddings: usize,
    embedding_dim: usize,
    batch: usize,
    heads: usize,
    seq_len: usize,
) -> AssertionCase {
    let embedding_data = layer_data(num_embeddings * embedding_dim, -0.6);
    let indices_data = index_data(batch * heads * seq_len, num_embeddings);
    fusor_conformance::assert(move |device: Device| {
        let embedding_data = embedding_data.clone();
        let indices_data = indices_data.clone();
        async move {
            let embeddings =
                Tensor::from_slice(&device, [num_embeddings, embedding_dim], &embedding_data);
            let indices = Tensor::from_slice(&device, [batch, heads, seq_len], &indices_data);
            let layer = Embedding::new_from_tensor(embeddings);
            let actual: Tensor<4, f32> = layer.forward(&indices);
            actual
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(exact_compare::<4, f32>())
    .runs(1)
    .into_case(format!(
        "layer_ops::embedding_lookup_matches_cpu_reference_on_varied_shapes::3d_{num_embeddings}x{embedding_dim}_{batch}x{heads}x{seq_len}"
    ))
}

fn assert_layer_norm_2d_case(batch: usize, features: usize, with_bias: bool) -> AssertionCase {
    let input_data = layer_data(batch * features, 0.15);
    let weight_data = layer_data(features, 1.0);
    let bias_data = with_bias.then(|| layer_data(features, -0.2));

    fusor_conformance::assert(move |device: Device| {
        let input_data = input_data.clone();
        let weight_data = weight_data.clone();
        let bias_data = bias_data.clone();
        async move {
            let weight = Tensor::from_slice(&device, [features], &weight_data);
            let bias = bias_data
                .as_ref()
                .map(|data| Tensor::from_slice(&device, [features], data));
            let layer_norm = LayerNorm::new(weight, bias, 1e-5);
            let input = Tensor::from_slice(&device, [batch, features], &input_data);
            layer_norm.forward(&input).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<2, f32>(1e-4))
    .runs(1)
    .into_case(format!(
        "layer_ops::layer_norm_matches_cpu_reference_on_varied_shapes::2d_b{batch}_f{features}_bias{with_bias}"
    ))
}

fn assert_layer_norm_3d_case(
    batch: usize,
    seq_len: usize,
    features: usize,
    with_bias: bool,
) -> AssertionCase {
    let input_data = layer_data(batch * seq_len * features, -0.4);
    let weight_data = layer_data(features, 0.8);
    let bias_data = with_bias.then(|| layer_data(features, 0.3));

    fusor_conformance::assert(move |device: Device| {
        let input_data = input_data.clone();
        let weight_data = weight_data.clone();
        let bias_data = bias_data.clone();
        async move {
            let weight = Tensor::from_slice(&device, [features], &weight_data);
            let bias = bias_data
                .as_ref()
                .map(|data| Tensor::from_slice(&device, [features], data));
            let layer_norm = LayerNorm::new(weight, bias, 1e-5);
            let input = Tensor::from_slice(&device, [batch, seq_len, features], &input_data);
            layer_norm.forward(&input).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(1)
    .into_case(format!(
        "layer_ops::layer_norm_matches_cpu_reference_on_varied_shapes::3d_b{batch}_s{seq_len}_f{features}_bias{with_bias}"
    ))
}

fn assert_rms_norm_2d_case(batch: usize, features: usize, with_bias: bool) -> AssertionCase {
    let input_data = layer_data(batch * features, 0.5);
    let weight_data = layer_data(features, 1.2);
    let bias_data = with_bias.then(|| layer_data(features, -0.3));

    fusor_conformance::assert(move |device: Device| {
        let input_data = input_data.clone();
        let weight_data = weight_data.clone();
        let bias_data = bias_data.clone();
        async move {
            let weight = Tensor::from_slice(&device, [features], &weight_data);
            let bias = bias_data
                .as_ref()
                .map(|data| Tensor::from_slice(&device, [features], data));
            let rms_norm = RmsNorm::new(weight, bias, 1e-5);
            let input = Tensor::from_slice(&device, [batch, features], &input_data);
            rms_norm.forward(&input).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<2, f32>(1e-4))
    .runs(1)
    .into_case(format!(
        "layer_ops::rms_norm_matches_cpu_reference_on_varied_shapes::2d_b{batch}_f{features}_bias{with_bias}"
    ))
}

fn assert_rms_norm_3d_case(
    batch: usize,
    seq_len: usize,
    features: usize,
    with_bias: bool,
) -> AssertionCase {
    let input_data = layer_data(batch * seq_len * features, -0.55);
    let weight_data = layer_data(features, 0.95);
    let bias_data = with_bias.then(|| layer_data(features, 0.2));

    fusor_conformance::assert(move |device: Device| {
        let input_data = input_data.clone();
        let weight_data = weight_data.clone();
        let bias_data = bias_data.clone();
        async move {
            let weight = Tensor::from_slice(&device, [features], &weight_data);
            let bias = bias_data
                .as_ref()
                .map(|data| Tensor::from_slice(&device, [features], data));
            let rms_norm = RmsNorm::new(weight, bias, 1e-5);
            let input = Tensor::from_slice(&device, [batch, seq_len, features], &input_data);
            rms_norm.forward(&input).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(1)
    .into_case(format!(
        "layer_ops::rms_norm_matches_cpu_reference_on_varied_shapes::3d_b{batch}_s{seq_len}_f{features}_bias{with_bias}"
    ))
}

fn assert_rms_norm_4d_case(
    batch: usize,
    heads: usize,
    seq_len: usize,
    features: usize,
    with_bias: bool,
) -> AssertionCase {
    let input_data = layer_data(batch * heads * seq_len * features, 0.7);
    let weight_data = layer_data(features, 1.1);
    let bias_data = with_bias.then(|| layer_data(features, -0.1));

    fusor_conformance::assert(move |device: Device| {
        let input_data = input_data.clone();
        let weight_data = weight_data.clone();
        let bias_data = bias_data.clone();
        async move {
            let weight = Tensor::from_slice(&device, [features], &weight_data);
            let bias = bias_data
                .as_ref()
                .map(|data| Tensor::from_slice(&device, [features], data));
            let rms_norm = RmsNorm::new(weight, bias, 1e-5);
            let input = Tensor::from_slice(&device, [batch, heads, seq_len, features], &input_data);
            rms_norm.forward(&input).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<4, f32>(1e-4))
    .runs(1)
    .into_case(format!(
        "layer_ops::rms_norm_matches_cpu_reference_on_varied_shapes::4d_b{batch}_h{heads}_s{seq_len}_f{features}_bias{with_bias}"
    ))
}

pub fn conv1d_matches_cpu_reference_on_varied_shapes() -> AssertionCases {
    [
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
    ]
    .into_iter()
    .map(assert_conv1d_case)
    .collect::<Vec<_>>()
    .into()
}

pub fn conv1d_properties_match_configuration() -> AssertionCases {
    [(2usize, 3usize, 1usize, 2usize, 3usize), (4, 2, 5, 1, 2)]
        .into_iter()
        .map(
            |(out_channels, in_channels, kernel_size, padding, stride)| {
                fusor_conformance::assert(move |device: Device| async move {
                    let weight = Tensor::from_slice(
                        &device,
                        [out_channels, in_channels, kernel_size],
                        &vec![0.0f32; out_channels * in_channels * kernel_size],
                    );
                    let conv = ConvNd::new(
                        weight,
                        None,
                        ConvNdConfig {
                            padding: [padding],
                            stride: [stride],
                            groups: 1,
                        },
                    );

                    (
                        conv.in_channels(),
                        conv.out_channels(),
                        conv.weight().shape()[2],
                        conv.config().padding[0],
                        conv.config().stride[0],
                    )
                })
                .arg(|device: &Device| device.clone())
                .equal_to(move |_device: Device| async move {
                    (in_channels, out_channels, kernel_size, padding, stride)
                })
                .compare_with(exact_value_compare())
                .runs(1)
                .into_case(format!(
                    "layer_ops::conv1d_properties_match_configuration::oc{out_channels}_ic{in_channels}_k{kernel_size}_pad{padding}_stride{stride}"
                ))
            },
        )
        .collect::<Vec<_>>()
        .into()
}

pub fn embedding_lookup_matches_cpu_reference_on_varied_shapes() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    assertions.extend(assert_embedding_1d_case(5, 3, 6));
    assertions.push(assert_embedding_2d_case(7, 4, 2, 5));
    assertions.push(assert_embedding_3d_case(6, 5, 2, 3, 4));
    assertions
}

pub fn layer_norm_matches_cpu_reference_on_varied_shapes() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for &(batch, features, with_bias) in &[(2usize, 3usize, false), (3, 5, true), (4, 7, true)] {
        assertions.push(assert_layer_norm_2d_case(batch, features, with_bias));
    }
    for &(batch, seq_len, features, with_bias) in &[
        (1usize, 2usize, 2usize, false),
        (2, 3, 4, true),
        (3, 2, 6, true),
    ] {
        assertions.push(assert_layer_norm_3d_case(
            batch, seq_len, features, with_bias,
        ));
    }
    assertions
}

pub fn layer_norm_fused_cpu_matches_reference_on_varied_shapes() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for &(batch, seq_len, features) in &[(1usize, 2usize, 2usize), (2, 3, 4), (1, 4, 7)] {
        let weight_data = layer_data(features, 1.3);
        let bias_data = layer_data(features, -0.45);
        let input_data = layer_data(batch * seq_len * features, 0.2);
        let reference_weight_data = weight_data.clone();
        let reference_bias_data = bias_data.clone();
        let reference_input_data = input_data.clone();

        assertions.push(
            fusor_conformance::assert(move |device: Device| {
                let weight_data = weight_data.clone();
                let bias_data = bias_data.clone();
                let input_data = input_data.clone();
                async move {
                    let weight = Tensor::from_slice(&device, [features], &weight_data);
                    let bias = Tensor::from_slice(&device, [features], &bias_data);
                    let layer_norm = LayerNorm::new(weight, Some(bias), 1e-5);
                    let input =
                        Tensor::from_slice(&device, [batch, seq_len, features], &input_data);
                    layer_norm.forward_fused(&input).to_concrete()
                }
            })
            .arg(|device: &Device| device.clone())
            .equal_to(move |device: Device| {
                let weight_data = reference_weight_data.clone();
                let bias_data = reference_bias_data.clone();
                let input_data = reference_input_data.clone();
                async move {
                    let weight = Tensor::from_slice(&device, [features], &weight_data);
                    let bias = Tensor::from_slice(&device, [features], &bias_data);
                    let layer_norm = LayerNorm::new(weight, Some(bias), 1e-5);
                    let input =
                        Tensor::from_slice(&device, [batch, seq_len, features], &input_data);
                    layer_norm.forward(&input).to_concrete()
                }
            })
            .compare_with(approx_compare::<3, f32>(1e-4))
            .devices([Device::Cpu])
            .runs(1)
            .into_case(format!(
                "layer_ops::layer_norm_fused_cpu_matches_reference_on_varied_shapes::b{batch}_s{seq_len}_f{features}"
            )),
        );
    }
    assertions
}

pub fn rms_norm_matches_cpu_reference_on_varied_shapes() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for &(batch, features, with_bias) in &[(2usize, 3usize, false), (3, 5, true), (4, 6, true)] {
        assertions.push(assert_rms_norm_2d_case(batch, features, with_bias));
    }
    for &(batch, seq_len, features, with_bias) in &[
        (1usize, 2usize, 2usize, false),
        (2, 3, 4, true),
        (3, 2, 5, true),
    ] {
        assertions.push(assert_rms_norm_3d_case(batch, seq_len, features, with_bias));
    }
    for &(batch, heads, seq_len, features, with_bias) in
        &[(1usize, 2usize, 3usize, 4usize, false), (2, 3, 2, 5, true)]
    {
        assertions.push(assert_rms_norm_4d_case(
            batch, heads, seq_len, features, with_bias,
        ));
    }
    assertions
}
