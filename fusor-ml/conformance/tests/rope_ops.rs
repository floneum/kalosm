mod common;

use common::{assert_approx_tensors, reshape4, rope_interleaved_4d, rope_normal_4d};
use fusor::{RopeCache, Tensor, ToVec1, base_inverse_frequency};
use fusor_conformance::{FuzzGenerator, GenerateFromDevice, available_devices};
use rand::distr::Uniform;

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

static COS: &[[f32; 2]] = &[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
static SIN: &[[f32; 2]] = &[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];

fn cos_vec() -> Vec<Vec<f32>> {
    COS.iter().map(|r| r.to_vec()).collect()
}
fn sin_vec() -> Vec<Vec<f32>> {
    SIN.iter().map(|r| r.to_vec()).collect()
}

#[tokio::test]
async fn rope_and_cache_paths_match_reference_variants() {
    let expected = vec![
        1.0,
        1.0 / 10_000.0f32.powf(2.0 / 8.0),
        1.0 / 10_000.0f32.powf(4.0 / 8.0),
        1.0 / 10_000.0f32.powf(6.0 / 8.0),
    ];
    assert_eq!(base_inverse_frequency(8, 10_000.0), expected);

    // Fuzz input tensor [1, 2, 3, 4] — cos/sin tables stay fixed
    // Note: ToVec is not implemented for rank-4, so we use available_devices loop
    // with fuzz generator for non-contiguous layout testing
    let mut fuzz_input = FuzzGenerator::<4, f32>::new([1, 2, 3, 4])
        .with_seed(600)
        .with_distribution(Uniform::new(-6.0, 6.0).unwrap());

    for device in available_devices().await {
        for run in 0..3 {
            let x = fuzz_input.generate(&device, run);
            let cos_t: Tensor<2, f32> = Tensor::new(&device, &cos_vec());
            let sin_t: Tensor<2, f32> = Tensor::new(&device, &sin_vec());

            // rope vs host reference
            let slice = x
                .to_concrete()
                .flatten_all()
                .to_concrete()
                .as_slice()
                .await
                .unwrap();
            let flat: Vec<f32> = slice.to_vec1();
            let v = reshape4(&flat, [1, 2, 3, 4]);
            let expected_normal = rope_normal_4d(&v, &cos_vec(), &sin_vec());
            assert_approx_tensors(
                x.rope(&cos_t, &sin_t),
                Tensor::new(&device, &expected_normal),
                1e-4,
            )
            .await;

            // rope_interleaved vs host reference
            let expected_interleaved = rope_interleaved_4d(&v, &cos_vec(), &sin_vec());
            assert_approx_tensors(
                x.rope_interleaved(&cos_t, &sin_t),
                Tensor::new(&device, &expected_interleaved),
                1e-4,
            )
            .await;

            // rope_normal_fused vs rope
            assert_approx_tensors(
                x.rope_normal_fused(&cos_t, &sin_t),
                x.rope(&cos_t, &sin_t),
                1e-4,
            )
            .await;

            // rope_fused vs rope_interleaved
            assert_approx_tensors(
                x.rope_fused(&cos_t, &sin_t),
                x.rope_interleaved(&cos_t, &sin_t),
                1e-4,
            )
            .await;
        }
    }

    // RopeCache tests
    for device in available_devices().await {
        let cache_cos = vec![
            vec![1.0f32, 1.1],
            vec![1.2, 1.3],
            vec![1.4, 1.5],
            vec![1.6, 1.7],
        ];
        let cache_sin = vec![
            vec![0.1f32, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6],
            vec![0.7, 0.8],
        ];
        let cache = RopeCache::from_parts(
            Tensor::new(&device, &cache_cos),
            Tensor::new(&device, &cache_sin),
        );

        let gen_q = FuzzGenerator::<4, f32>::new([1, 2, 3, 4])
            .with_seed(601)
            .with_distribution(Uniform::new(-6.0, 6.0).unwrap());
        let gen_k = FuzzGenerator::<4, f32>::new([1, 2, 3, 4])
            .with_seed(602)
            .with_distribution(Uniform::new(-6.0, 6.0).unwrap());

        let q = gen_q.clone().generate(&device, 0);
        let k = gen_k.clone().generate(&device, 0);

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
