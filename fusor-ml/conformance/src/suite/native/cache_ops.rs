//! Cache op conformance cases.

use fusor::{
    Device, Tensor,
    cache::{AttentionMask, KvCache, MaskCache, TensorCache},
};
use fusor_conformance::{
    AssertionCase, AssertionCases, available_devices, exact_compare, exact_value_compare,
};

type TensorSnapshot = (Vec<usize>, Vec<f32>);
type TensorCacheTrace = Vec<(TensorSnapshot, usize, bool)>;
type KvCacheTrace = Vec<(TensorSnapshot, TensorSnapshot, usize, bool, bool)>;

fn tensor_data(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i % 11) as f32) - 5.0) * 0.25 + offset)
        .collect()
}

fn index_iter<const R: usize>(shape: [usize; R]) -> impl Iterator<Item = [usize; R]> {
    let total: usize = shape.iter().product();
    (0..total).map(move |flat| {
        let mut idx = [0usize; R];
        let mut rem = flat;
        for dim in (0..R).rev() {
            idx[dim] = rem % shape[dim];
            rem /= shape[dim];
        }
        idx
    })
}

async fn tensor_snapshot<const R: usize>(tensor: Tensor<R, f32>) -> TensorSnapshot {
    let shape_array = tensor.shape();
    let shape = shape_array.to_vec();
    let slice = tensor.as_slice().await.unwrap();
    let values = index_iter(shape_array).map(|index| slice[index]).collect();
    (shape, values)
}

fn assert_attention_mask_apply_3d_case(batch: usize, seq_len: usize) -> AssertionCase {
    let scores_data = tensor_data(batch * seq_len * seq_len, 0.5);
    fusor_conformance::assert(move |device: Device| {
        let scores_data = scores_data.clone();
        async move {
            let mask: AttentionMask<f32> = AttentionMask::causal(&device, seq_len);
            let scores = Tensor::from_slice(&device, [batch, seq_len, seq_len], &scores_data);
            mask.apply(&scores).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(exact_compare::<3, f32>())
    .runs(1)
    .into_case(format!(
        "cache_ops::attention_mask_apply_broadcasts_to_varied_3d_and_4d_shapes::3d_{batch}x{seq_len}"
    ))
}

fn assert_attention_mask_apply_4d_case(
    batch: usize,
    heads: usize,
    seq_len: usize,
) -> AssertionCase {
    let scores_data = tensor_data(batch * heads * seq_len * seq_len, -0.25);
    fusor_conformance::assert(move |device: Device| {
        let scores_data = scores_data.clone();
        async move {
            let mask: AttentionMask<f32> = AttentionMask::causal(&device, seq_len);
            let scores =
                Tensor::from_slice(&device, [batch, heads, seq_len, seq_len], &scores_data);
            mask.apply(&scores).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(exact_compare::<4, f32>())
    .runs(1)
    .into_case(format!(
        "cache_ops::attention_mask_apply_broadcasts_to_varied_3d_and_4d_shapes::4d_{batch}x{heads}x{seq_len}"
    ))
}

pub fn attention_mask_causal_matches_expected_on_varied_sizes() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for seq_len in [1, 2, 4, 7] {
        assertions.push(
            fusor_conformance::assert(move |device: Device| async move {
                let mask: AttentionMask<f32> = AttentionMask::causal(&device, seq_len);
                mask.mask().clone()
            })
            .arg(|device: &Device| device.clone())
            .compare_with(exact_compare::<2, f32>())
            .runs(1)
            .into_case(format!(
                "cache_ops::attention_mask_causal_matches_expected_on_varied_sizes::{seq_len}"
            )),
        );
    }
    assertions
}

pub fn attention_mask_apply_broadcasts_to_varied_3d_and_4d_shapes() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for (batch, seq_len) in [(1, 2), (2, 3), (3, 4)] {
        assertions.push(assert_attention_mask_apply_3d_case(batch, seq_len));
    }
    for (batch, heads, seq_len) in [(1, 1, 2), (2, 3, 3), (2, 2, 5)] {
        assertions.push(assert_attention_mask_apply_4d_case(batch, heads, seq_len));
    }
    assertions
}

pub fn tensor_cache_append_and_reset_work_across_varied_cases() -> AssertionCases {
    let mut assertions = AssertionCases::new();

    for &(max_sequence_len, batch, features, chunk_lens) in &[
        (5usize, 1usize, 2usize, &[1usize, 2, 1][..]),
        (4usize, 2usize, 3usize, &[2usize, 2, 1][..]),
    ] {
        let chunk_lens = chunk_lens.to_vec();
        assertions.push(fusor_conformance::assert(move |device: Device| {
            let chunk_lens = chunk_lens.clone();
            async move {
                let mut cache: TensorCache<3, f32> = TensorCache::new(1, max_sequence_len);
                let mut trace = TensorCacheTrace::new();
                for (step, chunk_len) in chunk_lens.into_iter().enumerate() {
                    let data = tensor_data(batch * chunk_len * features, step as f32 + 0.25);
                    let tensor = Tensor::from_slice(&device, [batch, chunk_len, features], &data);
                    let appended = cache.append(&device, &tensor);
                    trace.push((
                        tensor_snapshot(appended).await,
                        cache.current_seq_len(),
                        cache.current_data().is_some(),
                    ));
                }
                cache.reset();
                trace.push((
                    (Vec::new(), Vec::new()),
                    cache.current_seq_len(),
                    cache.current_data().is_some(),
                ));
                trace
            }
        })
        .arg(|device: &Device| device.clone())
        .compare_with(exact_value_compare())
        .runs(1)
        .into_case(format!(
            "cache_ops::tensor_cache_append_and_reset_work_across_varied_cases::axis1_len{max_sequence_len}_batch{batch}_features{features}"
        )));
    }

    for &(max_sequence_len, batch, channels, chunk_lens) in &[
        (6usize, 1usize, 2usize, &[1usize, 3, 2][..]),
        (5usize, 2usize, 3usize, &[2usize, 1, 3][..]),
    ] {
        let chunk_lens = chunk_lens.to_vec();
        assertions.push(fusor_conformance::assert(move |device: Device| {
            let chunk_lens = chunk_lens.clone();
            async move {
                let mut cache: TensorCache<3, f32> = TensorCache::new(2, max_sequence_len);
                let mut trace = TensorCacheTrace::new();
                for (step, chunk_len) in chunk_lens.into_iter().enumerate() {
                    let data = tensor_data(batch * channels * chunk_len, step as f32 + 1.5);
                    let tensor = Tensor::from_slice(&device, [batch, channels, chunk_len], &data);
                    let appended = cache.append(&device, &tensor);
                    trace.push((
                        tensor_snapshot(appended).await,
                        cache.current_seq_len(),
                        cache.current_data().is_some(),
                    ));
                }
                trace
            }
        })
        .arg(|device: &Device| device.clone())
        .compare_with(exact_value_compare())
        .runs(1)
        .into_case(format!(
            "cache_ops::tensor_cache_append_and_reset_work_across_varied_cases::axis2_len{max_sequence_len}_batch{batch}_channels{channels}"
        )));
    }
    assertions
}

pub fn tensor_cache_gpu_lazy_appends_preserve_pending_writes() -> AssertionCase {
    fusor_conformance::assert(async |device: Device| {
        let mut cache: TensorCache<3, f32> = TensorCache::new(1, 8);
        let mut trace = TensorCacheTrace::new();
        for (step, &chunk_len) in [5usize, 1, 1].iter().enumerate() {
            let data = tensor_data(chunk_len * 2, step as f32 + 0.5);
            let tensor = Tensor::from_slice(&device, [1, chunk_len, 2], &data);
            let appended = cache.append(&device, &tensor);
            trace.push((
                tensor_snapshot(appended).await,
                cache.current_seq_len(),
                cache.current_data().is_some(),
            ));
        }
        trace
    })
    .arg(|device: &Device| device.clone())
    .compare_with(exact_value_compare())
    .devices_async(async {
        available_devices()
            .await
            .into_iter()
            .filter(Device::is_gpu)
            .collect()
    })
    .runs(1)
    .into_case("cache_ops::tensor_cache_gpu_lazy_appends_preserve_pending_writes")
}

pub fn kv_cache_append_and_reset_work_across_varied_cases() -> AssertionCases {
    let mut assertions = AssertionCases::new();

    for &(max_sequence_len, batch, heads, dim, chunk_lens) in &[
        (4usize, 1usize, 1usize, 2usize, &[1usize, 2, 1][..]),
        (5usize, 2usize, 3usize, 4usize, &[2usize, 1, 2][..]),
    ] {
        let chunk_lens = chunk_lens.to_vec();
        assertions.push(fusor_conformance::assert(move |device: Device| {
            let chunk_lens = chunk_lens.clone();
            async move {
                let mut cache: KvCache<f32> = KvCache::new(1, max_sequence_len);
                let mut trace = KvCacheTrace::new();

                for (step, chunk_len) in chunk_lens.into_iter().enumerate() {
                    let key_data = tensor_data(batch * chunk_len * heads * dim, step as f32 + 0.75);
                    let value_data =
                        tensor_data(batch * chunk_len * heads * dim, step as f32 + 2.25);
                    let key = Tensor::from_slice(&device, [batch, chunk_len, heads, dim], &key_data);
                    let value =
                        Tensor::from_slice(&device, [batch, chunk_len, heads, dim], &value_data);

                    let (keys, values) = cache.append(&device, &key, &value);
                    trace.push((
                        tensor_snapshot(keys).await,
                        tensor_snapshot(values).await,
                        cache.current_seq_len(),
                        cache.k().is_some(),
                        cache.v().is_some(),
                    ));
                }

                cache.reset();
                trace.push((
                    (Vec::new(), Vec::new()),
                    (Vec::new(), Vec::new()),
                    cache.current_seq_len(),
                    cache.k().is_some(),
                    cache.v().is_some(),
                ));
                trace
            }
        })
        .arg(|device: &Device| device.clone())
        .compare_with(exact_value_compare())
        .runs(1)
        .into_case(format!(
            "cache_ops::kv_cache_append_and_reset_work_across_varied_cases::len{max_sequence_len}_batch{batch}_heads{heads}_dim{dim}"
        )));
    }
    assertions
}

pub fn mask_cache_supports_varied_offsets_and_sliding_windows() -> AssertionCases {
    let mut assertions = AssertionCases::new();

    for &(seq_len, offset, sliding_window) in &[
        (1usize, 0usize, None),
        (3usize, 2usize, None),
        (4usize, 0usize, Some(2usize)),
        (5usize, 3usize, Some(4usize)),
    ] {
        assertions.push(fusor_conformance::assert(move |device: Device| async move {
            let cache: MaskCache<f32> = MaskCache::default();
            cache
                .get_mask(seq_len, offset, sliding_window, &device)
                .mask()
                .clone()
        })
        .arg(|device: &Device| device.clone())
        .compare_with(exact_compare::<2, f32>())
        .runs(1)
        .into_case(format!(
            "cache_ops::mask_cache_supports_varied_offsets_and_sliding_windows::seq{seq_len}_offset{offset}_window{sliding_window:?}"
        )));
    }
    assertions
}
