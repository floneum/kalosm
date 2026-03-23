mod common;

use common::{
    assert_approx_devices, assert_approx_tensors, assert_exact_tensors, reshape4,
    rope_interleaved_4d, rope_normal_4d,
};
use fusor::{RopeCache, Tensor, base_inverse_frequency};
use fusor_conformance::available_devices;

fn rope_input() -> Vec<Vec<Vec<Vec<f32>>>> {
    reshape4(
        &(0..24).map(|value| value as f32 - 6.0).collect::<Vec<_>>(),
        [1, 2, 3, 4],
    )
}

fn rope_cos() -> Vec<Vec<f32>> {
    vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
}

fn rope_sin() -> Vec<Vec<f32>> {
    vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]]
}

fn rope_cache_cos() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 1.1],
        vec![1.2, 1.3],
        vec![1.4, 1.5],
        vec![1.6, 1.7],
    ]
}

fn rope_cache_sin() -> Vec<Vec<f32>> {
    vec![
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
        vec![0.7, 0.8],
    ]
}

fn rope_tables(
    head_dim: usize,
    context_length: usize,
    theta: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let inv_freq = base_inverse_frequency(head_dim, theta);
    let cos = (0..context_length)
        .map(|position| {
            inv_freq
                .iter()
                .map(|freq| (position as f32 * freq).cos())
                .collect()
        })
        .collect();
    let sin = (0..context_length)
        .map(|position| {
            inv_freq
                .iter()
                .map(|freq| (position as f32 * freq).sin())
                .collect()
        })
        .collect();
    (cos, sin)
}

#[tokio::test]
async fn rope_and_cache_paths_match_reference_variants() {
    let input = rope_input();
    let cos = rope_cos();
    let sin = rope_sin();

    let expected = vec![
        1.0,
        1.0 / 10_000.0f32.powf(2.0 / 8.0),
        1.0 / 10_000.0f32.powf(4.0 / 8.0),
        1.0 / 10_000.0f32.powf(6.0 / 8.0),
    ];
    assert_eq!(base_inverse_frequency(8, 10_000.0), expected);

    assert_approx_devices(
        |device| {
            let x: Tensor<4, f32> = Tensor::new(device, &input);
            let cos: Tensor<2, f32> = Tensor::new(device, &cos);
            let sin: Tensor<2, f32> = Tensor::new(device, &sin);
            x.rope(&cos, &sin)
        },
        |device| Tensor::new(device, &rope_normal_4d(&input, &cos, &sin)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| {
            let x: Tensor<4, f32> = Tensor::new(device, &input);
            let cos: Tensor<2, f32> = Tensor::new(device, &cos);
            let sin: Tensor<2, f32> = Tensor::new(device, &sin);
            x.rope_interleaved(&cos, &sin)
        },
        |device| Tensor::new(device, &rope_interleaved_4d(&input, &cos, &sin)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| {
            let x: Tensor<4, f32> = Tensor::new(device, &input);
            let cos: Tensor<2, f32> = Tensor::new(device, &cos);
            let sin: Tensor<2, f32> = Tensor::new(device, &sin);
            x.rope_normal_fused(&cos, &sin)
        },
        |device| {
            let x: Tensor<4, f32> = Tensor::new(device, &input);
            let cos: Tensor<2, f32> = Tensor::new(device, &cos);
            let sin: Tensor<2, f32> = Tensor::new(device, &sin);
            x.rope(&cos, &sin)
        },
        1e-4,
    )
    .await;

    assert_approx_devices(
        |device| {
            let x: Tensor<4, f32> = Tensor::new(device, &input);
            let cos: Tensor<2, f32> = Tensor::new(device, &cos);
            let sin: Tensor<2, f32> = Tensor::new(device, &sin);
            x.rope_fused(&cos, &sin)
        },
        |device| {
            let x: Tensor<4, f32> = Tensor::new(device, &input);
            let cos: Tensor<2, f32> = Tensor::new(device, &cos);
            let sin: Tensor<2, f32> = Tensor::new(device, &sin);
            x.rope_interleaved(&cos, &sin)
        },
        1e-4,
    )
    .await;

    for device in available_devices().await {
        let cache_cos = rope_cache_cos();
        let cache_sin = rope_cache_sin();
        let cache = RopeCache::from_parts(
            Tensor::new(&device, &cache_cos),
            Tensor::new(&device, &cache_sin),
        );
        assert_exact_tensors(cache.cos().clone(), Tensor::new(&device, &cache_cos)).await;
        assert_exact_tensors(cache.sin().clone(), Tensor::new(&device, &cache_sin)).await;

        let q: Tensor<4, f32> = Tensor::new(&device, &input);
        let k: Tensor<4, f32> = Tensor::new(
            &device,
            &reshape4(
                &(10..34).map(|value| value as f32 - 8.0).collect::<Vec<_>>(),
                [1, 2, 3, 4],
            ),
        );
        let cos_slice: Tensor<2, f32> = Tensor::new(&device, &cache_cos[1..4].to_vec());
        let sin_slice: Tensor<2, f32> = Tensor::new(&device, &cache_sin[1..4].to_vec());

        let (q_rope, k_rope) = cache.forward(&q, &k, 1);
        assert_approx_tensors(q_rope, q.rope_normal_fused(&cos_slice, &sin_slice), 1e-4).await;
        assert_approx_tensors(k_rope, k.rope_normal_fused(&cos_slice, &sin_slice), 1e-4).await;

        let (q_interleaved, k_interleaved) = cache.forward_interleaved(&q, &k, 1);
        assert_approx_tensors(q_interleaved, q.rope_fused(&cos_slice, &sin_slice), 1e-4).await;
        assert_approx_tensors(k_interleaved, k.rope_fused(&cos_slice, &sin_slice), 1e-4).await;

        let built_cache = RopeCache::new(4, 3, 10_000.0, &device).unwrap();
        let (expected_cos, expected_sin) = rope_tables(4, 3, 10_000.0);
        assert_approx_tensors(
            built_cache.cos().clone(),
            Tensor::new(&device, &expected_cos),
            1e-6,
        )
        .await;
        assert_approx_tensors(
            built_cache.sin().clone(),
            Tensor::new(&device, &expected_sin),
            1e-6,
        )
        .await;
    }
}
