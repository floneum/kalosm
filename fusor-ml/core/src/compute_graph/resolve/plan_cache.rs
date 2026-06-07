//! Structural plan-cache keys for resolved operations.

use crate::mir::inputs::MirValue;
use crate::mir::kernel_backend::{KernelCacheKey, KernelVariantKey};
use crate::mir::operation::Operation;
use crate::mir::workgroup_shape::WorkgroupShape;

struct DirectPlanCacheKernelVariant;

pub(crate) fn structural_kernel_key(
    operation: &dyn Operation,
    inputs: &[MirValue],
    workgroup: &WorkgroupShape,
) -> KernelCacheKey {
    let dispatch_size = operation.dispatch_size(workgroup, inputs);
    operation.kernel_cache_key_with_dispatch(
        KernelVariantKey::of::<DirectPlanCacheKernelVariant>(),
        Some(workgroup),
        dispatch_size,
        inputs,
    )
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::{Device, QMatrix, Tensor};
    use fusor_gguf::GgmlType;

    const N: usize = 4;
    const K: usize = 8;

    // An F32 (native) quantized matrix [N, K] so `q_mat_mul` is a plain matmul.
    fn weight(device: &Device) -> QMatrix {
        let bytes: Vec<u8> = (0..N * K)
            .map(|i| 0.1 + (i as f32) * 0.05)
            .flat_map(f32::to_le_bytes)
            .collect();
        QMatrix::from_parts(device, &bytes, vec![N, K].into_boxed_slice(), GgmlType::F32).unwrap()
    }

    fn bias(device: &Device) -> Tensor {
        Tensor::new::<f32, 2, _>(device, &[[0.5f32, -1.0, 2.0, -0.25]])
    }

    async fn read_rows(out: &Tensor) -> Vec<f32> {
        let slice = out.as_slice::<2, f32>().await.unwrap();
        let shape = slice.shape();
        let (rows, cols) = (shape[0], shape[1]);
        let mut values = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                values.push(slice[[r, c]]);
            }
        }
        values
    }

    // Exercises the qmatmul build arm: the trailing add keeps the resolve target
    // off the single-qmatmul fast path, so the resolver (and the plan cache) runs.
    async fn run_qmatmul(device: &Device, w: &QMatrix, input: [f32; K]) -> Vec<f32> {
        let out = Tensor::new::<f32, 2, _>(device, &[input]).q_mat_mul(w);
        read_rows(&(&out + &bias(device))).await
    }

    // Exercises the generic (nary) build arm with two distinct tensor inputs.
    async fn run_nary(device: &Device, input: [f32; K]) -> Vec<f32> {
        let row: [f32; N] = input[..N].try_into().unwrap();
        let a = Tensor::new::<f32, 2, _>(device, &[row]);
        read_rows(&(&a + &bias(device))).await
    }

    async fn run_pairwise_add(device: &Device) -> Vec<f32> {
        let a = Tensor::new::<f32, 2, _>(device, &[[1.0f32, 2.0, 3.0, 4.0]]);
        let b = Tensor::new::<f32, 2, _>(device, &[[0.5f32, -1.0, 2.0, -0.25]]);
        read_rows(&(&a + &b)).await
    }

    async fn run_pairwise_self_add(device: &Device) -> Vec<f32> {
        let a = Tensor::new::<f32, 2, _>(device, &[[1.0f32, 2.0, 3.0, 4.0]]);
        read_rows(&(&a + &a)).await
    }

    async fn run_pairwise_mul(device: &Device) -> Vec<f32> {
        let a = Tensor::new::<f32, 2, _>(device, &[[1.0f32, 2.0, 3.0, 4.0]]);
        let b = Tensor::new::<f32, 2, _>(device, &[[0.5f32, -1.0, 2.0, -0.25]]);
        read_rows(&(&a * &b)).await
    }

    async fn run_slice_assign(device: &Device, slices: [Range<usize>; 2]) -> Vec<f32> {
        let base = Tensor::new::<f32, 2, _>(
            device,
            &[
                [0.0f32, 0.0, 0.0, 0.0],
                [0.0f32, 0.0, 0.0, 0.0],
                [0.0f32, 0.0, 0.0, 0.0],
            ],
        );
        let value = Tensor::new::<f32, 2, _>(device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        read_rows(&base.slice_assign(slices, &value)).await
    }

    // Exercises the `Sequence` kernel arm (scratch + reduce + write passes). A
    // large softmax takes the split path; candidate direct-plan bindings are not
    // inserted because scratch bindings are allocated inside the build.
    // `seed` makes the input (and output) input-dependent.
    async fn run_softmax(device: &Device, seed: f32) -> Vec<f32> {
        let data: Vec<f32> = (0..4096)
            .map(|i| ((i as f32) * 0.001 + seed).sin())
            .collect();
        let out = Tensor::new(device, data.as_slice()).softmax(0);
        let slice = out.as_slice::<1, f32>().await.unwrap();
        let len = slice.shape()[0];
        (0..len).step_by(257).map(|i| slice[[i]]).collect()
    }

    fn assert_close(a: &[f32], b: &[f32], context: &str) {
        assert_eq!(
            a.len(),
            b.len(),
            "{context}: length mismatch ({a:?} vs {b:?})"
        );
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= 1e-3 + 1e-3 * x.abs().max(y.abs()),
                "{context}: element {i} differs: {x} vs {y}"
            );
        }
    }

    // The rebind-on-cache-hit replay must (a) byte-match the cold build for the
    // same input, and (b) for a different input, match a cold build on a fresh
    // device (whose plan cache is empty, forcing the normal build path).
    // `device2` shares the same GPU, so the normal-path results are the golden
    // reference. Covers both build arms, including the fused-epilogue qmatmul the
    // old fast cache could not key.
    #[test]
    fn direct_plan_cache_rebind_matches_fresh_build() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let Ok(golden) = Device::new().await else {
                return;
            };

            let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let u = [-2.0, 1.5, 0.0, 3.0, -1.0, 2.5, 4.0, -0.5];

            // --- qmatmul arm ---
            let w = weight(&device);
            let w_golden = weight(&golden);
            let miss_v = run_qmatmul(&device, &w, v).await; // cold build
            let hit_v = run_qmatmul(&device, &w, v).await; // rebind replay
            assert_eq!(
                miss_v, hit_v,
                "qmatmul: rebind must byte-match the cold build"
            );
            let hit_u = run_qmatmul(&device, &w, u).await; // rebind, new buffers
            let golden_u = run_qmatmul(&golden, &w_golden, u).await; // cold build, fresh cache
            assert_close(&hit_u, &golden_u, "qmatmul: rebind with new buffers");
            assert!(hit_u != hit_v, "qmatmul: different inputs must differ");

            // --- generic (nary) arm ---
            let miss_v = run_nary(&device, v).await;
            let hit_v = run_nary(&device, v).await;
            assert_eq!(miss_v, hit_v, "nary: rebind must byte-match the cold build");
            let hit_u = run_nary(&device, u).await;
            let golden_u = run_nary(&golden, u).await;
            assert_close(&hit_u, &golden_u, "nary: rebind with new buffers");
            assert!(hit_u != hit_v, "nary: different inputs must differ");

            // --- Sequence arm (split softmax, not direct-plan cached yet) ---
            let miss_v = run_softmax(&device, 0.0).await;
            let repeat_v = run_softmax(&device, 0.0).await;
            assert_eq!(
                miss_v, repeat_v,
                "softmax: repeat build must byte-match the cold build"
            );
            let next_u = run_softmax(&device, 0.7).await;
            let golden_u = run_softmax(&golden, 0.7).await;
            assert_close(&next_u, &golden_u, "softmax: rebuild with new buffers");
            assert!(next_u != repeat_v, "softmax: different inputs must differ");
        });
    }

    #[test]
    fn direct_plan_cache_distinguishes_same_shape_generic_ops() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let Ok(golden) = Device::new().await else {
                return;
            };

            let add = run_pairwise_add(&device).await;
            let mul = run_pairwise_mul(&device).await;
            let golden_mul = run_pairwise_mul(&golden).await;

            assert_close(&mul, &golden_mul, "generic op key: multiply after add");
            assert!(mul != add, "generic op key: multiply must not replay add");
        });
    }

    #[test]
    fn direct_plan_cache_rebinds_repeated_binding_slots_positionally() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let Ok(golden) = Device::new().await else {
                return;
            };

            let alias_add = run_pairwise_self_add(&device).await;
            let distinct_add = run_pairwise_add(&device).await;
            let golden_distinct_add = run_pairwise_add(&golden).await;

            assert_close(
                &distinct_add,
                &golden_distinct_add,
                "repeated binding slots: distinct add after self add",
            );
            assert!(
                distinct_add != alias_add,
                "repeated binding slots: distinct add must not reuse the repeated buffer"
            );
        });
    }

    #[test]
    fn direct_plan_cache_distinguishes_same_shape_slice_assign_ranges() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let Ok(golden) = Device::new().await else {
                return;
            };

            let first = run_slice_assign(&device, [0..2, 0..2]).await;
            let second = run_slice_assign(&device, [1..3, 1..3]).await;
            let golden_second = run_slice_assign(&golden, [1..3, 1..3]).await;

            assert_close(
                &second,
                &golden_second,
                "slice_assign key: shifted range after top-left range",
            );
            assert!(
                second != first,
                "slice_assign key: shifted range must not replay top-left range"
            );
        });
    }
}
