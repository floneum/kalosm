//! The canonical structural folds behind the resolver's cache keys.
//!
//! The single-operation plan key and the horizontally merged plan key fold
//! the same item — an operation's type and kernel fields plus the MIR values
//! it binds — and differ only in how much dispatch identity survives, which
//! [`Identity`] names: a merged segment shares one grid the merged builder
//! derives from the whole wave, so per-segment geometry cannot key it.
//!
//! These keys select cached kernel plans that are replayed by positional
//! rebind and shared across processes through the persistent plan store, so
//! the exact bytes are load-bearing; `key_goldens` pins them.

use std::hash::Hash;

use rustc_hash::FxHasher;

use super::merge_horizontal::MergedSegments;
use crate::mir::inputs::MirValue;
use crate::mir::kernel_backend::{KernelCacheKey, KernelVariantKey};
use crate::mir::operation::{Operation, hash_mir_value};
use crate::mir::workgroup_shape::WorkgroupShape;

/// A variant marker's `TypeId` is hashed into every key it stamps, so its
/// declaration site — module path included — is part of the recipe.
struct KernelPlanCacheVariant;

/// How much of the dispatch an operation's structural key keeps.
enum Identity<'a> {
    /// The exact dispatch this operation will run: its solved workgroup
    /// shape and grid are baked into the generated kernel.
    Dispatch(&'a WorkgroupShape),
    /// Dispatch geometry erased, for segments whose grid is decided by the
    /// merged builder rather than by the segment itself.
    Erased,
}

/// One work item's structural key at the requested dispatch identity.
fn item_key(
    operation: &dyn Operation,
    identity: Identity<'_>,
    inputs: &[MirValue],
    variant: KernelVariantKey,
) -> KernelCacheKey {
    let (workgroup, dispatch_size) = match identity {
        Identity::Dispatch(workgroup) => {
            (Some(workgroup), operation.dispatch_size(workgroup, inputs))
        }
        Identity::Erased => (None, [0; 3]),
    };
    operation.kernel_cache_key_with_dispatch(variant, workgroup, dispatch_size, inputs)
}

pub(crate) fn structural_kernel_key(
    operation: &dyn Operation,
    inputs: &[MirValue],
    workgroup: &WorkgroupShape,
) -> KernelCacheKey {
    let operation_key = item_key(
        operation,
        Identity::Dispatch(workgroup),
        inputs,
        KernelVariantKey::of::<KernelPlanCacheVariant>(),
    );
    KernelCacheKey::from_hash_inputs(|state| {
        operation_key.hash(state);
    })
}

/// A structural plan-cache key for one horizontally merged dispatch: the
/// wave discriminant plus every segment's own structural key, so isomorphic
/// waves across resolves and processes share one plan. Region segments merge
/// without the `Operation` trait and fold their kernel fields inline.
pub(super) fn merged_segments_key(
    variant: KernelVariantKey,
    merged: &MergedSegments,
    segment_inputs: &[Vec<MirValue>],
) -> KernelCacheKey {
    KernelCacheKey::from_hash_inputs(|state| {
        variant.hash(state);
        std::mem::discriminant(merged).hash(state);
        match merged {
            MergedSegments::Region(segments) => {
                hash_merged_segments(state, segments.iter().map(|(_, op)| op), segment_inputs)
            }
            _ => {
                segment_inputs.len().hash(state);
                for ((_, op), inputs) in merged.segment_ops().iter().zip(segment_inputs) {
                    item_key(*op, Identity::Erased, inputs, variant).hash(state);
                }
            }
        }
    })
}

/// The kernel-field surface merged segments key on. Region segments merge
/// without the `Operation` trait, so the fold names this surface directly.
pub(crate) trait SegmentFields {
    fn hash_kernel_fields(&self, state: &mut FxHasher);
}

impl SegmentFields for crate::matmul::MatMulOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        Operation::hash_kernel_fields(self, state);
    }
}

impl SegmentFields for crate::row_program::RowProgramOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        Operation::hash_kernel_fields(self, state);
    }
}

impl SegmentFields for crate::region::ElementwiseRegionOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        crate::region::ElementwiseRegionOperation::hash_kernel_fields(self, state);
    }
}

/// Hash one merged dispatch's cache-key material: every segment's kernel
/// fields plus every MIR input value layout.
pub(crate) fn hash_merged_segments<'a, S: SegmentFields + 'a>(
    state: &mut FxHasher,
    segments: impl ExactSizeIterator<Item = &'a S>,
    segment_inputs: &[Vec<MirValue>],
) {
    segments.len().hash(state);
    for (op, inputs) in segments.zip(segment_inputs) {
        op.hash_kernel_fields(state);
        inputs.len().hash(state);
        for input in inputs {
            hash_mir_value(state, input);
        }
    }
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
    fn kernel_plan_cache_rebind_matches_fresh_build() {
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
    fn kernel_plan_cache_distinguishes_same_shape_generic_ops() {
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
    fn kernel_plan_cache_rebinds_repeated_binding_slots_positionally() {
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
    fn kernel_plan_cache_distinguishes_same_shape_slice_assign_ranges() {
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
