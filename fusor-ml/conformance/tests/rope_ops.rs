mod common;

use common::{reshape4, rope_interleaved_4d, rope_normal_4d};
use fusor::{Device, RopeCache, Tensor, ToVec1, base_inverse_frequency};
use fusor_conformance::{FuzzGenerator, approx_compare};
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

#[derive(Clone, Copy)]
enum RopeReference {
    Normal,
    Interleaved,
}

#[derive(Clone, Copy)]
enum RopeTensorOp {
    Normal,
    Interleaved,
    NormalFused,
    InterleavedFused,
}

#[derive(Clone, Copy)]
enum CacheForward {
    Normal,
    Interleaved,
}

#[derive(Clone, Copy)]
enum CacheOutput {
    Query,
    Key,
}

async fn rope_reference_tensor(
    x: Tensor<4, f32>,
    cos: Vec<Vec<f32>>,
    sin: Vec<Vec<f32>>,
    reference: RopeReference,
) -> Tensor<4, f32> {
    let device = x.device();
    let slice = x
        .to_concrete()
        .flatten_all()
        .to_concrete()
        .as_slice()
        .await
        .unwrap();
    let flat: Vec<f32> = slice.to_vec1();
    let host = reshape4(&flat, [1, 2, 3, 4]);
    let out = match reference {
        RopeReference::Normal => rope_normal_4d(&host, &cos, &sin),
        RopeReference::Interleaved => rope_interleaved_4d(&host, &cos, &sin),
    };
    Tensor::new(&device, &out)
}

fn apply_rope_op(
    x: Tensor<4, f32>,
    cos: Vec<Vec<f32>>,
    sin: Vec<Vec<f32>>,
    op: RopeTensorOp,
) -> Tensor<4, f32> {
    let device = x.device();
    let cos_t: Tensor<2, f32> = Tensor::new(&device, &cos);
    let sin_t: Tensor<2, f32> = Tensor::new(&device, &sin);
    match op {
        RopeTensorOp::Normal => x.rope(&cos_t, &sin_t),
        RopeTensorOp::Interleaved => x.rope_interleaved(&cos_t, &sin_t),
        RopeTensorOp::NormalFused => x.rope_normal_fused(&cos_t, &sin_t),
        RopeTensorOp::InterleavedFused => x.rope_fused(&cos_t, &sin_t),
    }
}

fn cache_forward_output(
    q: Tensor<4, f32>,
    k: Tensor<4, f32>,
    cos: Vec<Vec<f32>>,
    sin: Vec<Vec<f32>>,
    forward: CacheForward,
    output: CacheOutput,
) -> Tensor<4, f32> {
    let device = q.device();
    let cache = RopeCache::from_parts(Tensor::new(&device, &cos), Tensor::new(&device, &sin));
    let (q_out, k_out) = match forward {
        CacheForward::Normal => cache.forward(&q, &k, 1),
        CacheForward::Interleaved => cache.forward_interleaved(&q, &k, 1),
    };
    match output {
        CacheOutput::Query => q_out,
        CacheOutput::Key => k_out,
    }
}

fn cache_expected_output(
    q: Tensor<4, f32>,
    k: Tensor<4, f32>,
    cos: Vec<Vec<f32>>,
    sin: Vec<Vec<f32>>,
    output: CacheOutput,
    op: RopeTensorOp,
) -> Tensor<4, f32> {
    let cos_slice = cos[1..4].to_vec();
    let sin_slice = sin[1..4].to_vec();
    match output {
        CacheOutput::Query => apply_rope_op(q, cos_slice, sin_slice, op),
        CacheOutput::Key => apply_rope_op(k, cos_slice, sin_slice, op),
    }
}

macro_rules! assert_rope_matches_reference {
    ($fuzz:expr, $cos:expr, $sin:expr, $reference:expr, $op:expr) => {
        fusor_conformance::assert({
            let cos = $cos.clone();
            let sin = $sin.clone();
            move |x: Tensor<4, f32>| {
                let cos = cos.clone();
                let sin = sin.clone();
                async move { rope_reference_tensor(x, cos, sin, $reference).await }
            }
        })
        .arg($fuzz.clone())
        .equal_to({
            let cos = $cos.clone();
            let sin = $sin.clone();
            move |x: Tensor<4, f32>| {
                let cos = cos.clone();
                let sin = sin.clone();
                async move { apply_rope_op(x, cos, sin, $op) }
            }
        })
        .compare_with(approx_compare::<4, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();
    };
}

macro_rules! assert_rope_ops_match {
    ($fuzz:expr, $cos:expr, $sin:expr, $actual:expr, $expected:expr) => {
        fusor_conformance::assert({
            let cos = $cos.clone();
            let sin = $sin.clone();
            move |x: Tensor<4, f32>| {
                let cos = cos.clone();
                let sin = sin.clone();
                async move { apply_rope_op(x, cos, sin, $actual) }
            }
        })
        .arg($fuzz.clone())
        .equal_to({
            let cos = $cos.clone();
            let sin = $sin.clone();
            move |x: Tensor<4, f32>| {
                let cos = cos.clone();
                let sin = sin.clone();
                async move { apply_rope_op(x, cos, sin, $expected) }
            }
        })
        .compare_with(approx_compare::<4, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();
    };
}

macro_rules! assert_cache_path_matches_direct_rope {
    ($gen_q:expr, $gen_k:expr, $cos:expr, $sin:expr, $forward:expr, $output:expr, $expected:expr) => {
        fusor_conformance::assert({
            let cos = $cos.clone();
            let sin = $sin.clone();
            move |q: Tensor<4, f32>, k: Tensor<4, f32>| {
                let cos = cos.clone();
                let sin = sin.clone();
                async move { cache_forward_output(q, k, cos, sin, $forward, $output) }
            }
        })
        .arg($gen_q.clone())
        .arg($gen_k.clone())
        .equal_to({
            let cos = $cos.clone();
            let sin = $sin.clone();
            move |q: Tensor<4, f32>, k: Tensor<4, f32>| {
                let cos = cos.clone();
                let sin = sin.clone();
                async move { cache_expected_output(q, k, cos, sin, $output, $expected) }
            }
        })
        .compare_with(approx_compare::<4, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();
    };
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

    let cos = cos_vec();
    let sin = sin_vec();
    let fuzz_input = FuzzGenerator::<4, f32>::new([1, 2, 3, 4])
        .with_seed(600)
        .with_distribution(Uniform::new(-6.0, 6.0).unwrap());

    assert_rope_matches_reference!(
        fuzz_input,
        cos,
        sin,
        RopeReference::Normal,
        RopeTensorOp::Normal
    );
    assert_rope_matches_reference!(
        fuzz_input,
        cos,
        sin,
        RopeReference::Interleaved,
        RopeTensorOp::Interleaved
    );
    assert_rope_ops_match!(
        fuzz_input,
        cos,
        sin,
        RopeTensorOp::NormalFused,
        RopeTensorOp::Normal
    );
    assert_rope_ops_match!(
        fuzz_input,
        cos,
        sin,
        RopeTensorOp::InterleavedFused,
        RopeTensorOp::Interleaved
    );

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
    let gen_q = FuzzGenerator::<4, f32>::new([1, 2, 3, 4])
        .with_seed(601)
        .with_distribution(Uniform::new(-6.0, 6.0).unwrap());
    let gen_k = FuzzGenerator::<4, f32>::new([1, 2, 3, 4])
        .with_seed(602)
        .with_distribution(Uniform::new(-6.0, 6.0).unwrap());

    assert_cache_path_matches_direct_rope!(
        gen_q,
        gen_k,
        cache_cos,
        cache_sin,
        CacheForward::Normal,
        CacheOutput::Query,
        RopeTensorOp::NormalFused
    );
    assert_cache_path_matches_direct_rope!(
        gen_q,
        gen_k,
        cache_cos,
        cache_sin,
        CacheForward::Normal,
        CacheOutput::Key,
        RopeTensorOp::NormalFused
    );
    assert_cache_path_matches_direct_rope!(
        gen_q,
        gen_k,
        cache_cos,
        cache_sin,
        CacheForward::Interleaved,
        CacheOutput::Query,
        RopeTensorOp::InterleavedFused
    );
    assert_cache_path_matches_direct_rope!(
        gen_q,
        gen_k,
        cache_cos,
        cache_sin,
        CacheForward::Interleaved,
        CacheOutput::Key,
        RopeTensorOp::InterleavedFused
    );

    let gen_q_gqa = FuzzGenerator::<4, f32>::new([1, 4, 3, 4])
        .with_seed(603)
        .with_distribution(Uniform::new(-6.0, 6.0).unwrap());
    let gen_k_gqa = FuzzGenerator::<4, f32>::new([1, 2, 3, 4])
        .with_seed(604)
        .with_distribution(Uniform::new(-6.0, 6.0).unwrap());

    assert_cache_path_matches_direct_rope!(
        gen_q_gqa,
        gen_k_gqa,
        cache_cos,
        cache_sin,
        CacheForward::Normal,
        CacheOutput::Query,
        RopeTensorOp::NormalFused
    );
    assert_cache_path_matches_direct_rope!(
        gen_q_gqa,
        gen_k_gqa,
        cache_cos,
        cache_sin,
        CacheForward::Interleaved,
        CacheOutput::Key,
        RopeTensorOp::InterleavedFused
    );

    fusor_conformance::assert(async |device: Device| {
        RopeCache::new(4, 3, 10_000.0, &device)
            .unwrap()
            .cos()
            .clone()
    })
    .arg(|device: &Device| device.clone())
    .equal_to({
        let expected_cos = rope_tables(4, 3, 10_000.0).0;
        move |device: Device| {
            let expected_cos = expected_cos.clone();
            async move { Tensor::new(&device, &expected_cos) }
        }
    })
    .compare_with(approx_compare::<2, f32>(1e-6))
    .await
    .unwrap();

    fusor_conformance::assert(async |device: Device| {
        RopeCache::new(4, 3, 10_000.0, &device)
            .unwrap()
            .sin()
            .clone()
    })
    .arg(|device: &Device| device.clone())
    .equal_to({
        let expected_sin = rope_tables(4, 3, 10_000.0).1;
        move |device: Device| {
            let expected_sin = expected_sin.clone();
            async move { Tensor::new(&device, &expected_sin) }
        }
    })
    .compare_with(approx_compare::<2, f32>(1e-6))
    .await
    .unwrap();
}
