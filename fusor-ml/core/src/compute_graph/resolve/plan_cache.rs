//! Decode plan cache.
//!
//! During autoregressive decode, every token resolves a structurally identical
//! compute graph: the same operations, in the same toposorted order, with the
//! same shapes (only the KV-cache-dependent attention ops and the live buffer
//! handles change token-to-token). Yet `build_kernel` re-runs the full per-op
//! analysis — kernel-variant selection, workgroup solving, epilogue hashing,
//! tile-IR lowering checks — every token, even though the compiled pipelines
//! are already cached. On the web build that analysis dominates the per-token
//! host cost (~10ms of a ~16ms token).
//!
//! This cache memoizes bufferless [`DirectKernelTemplate`]s per operation and,
//! on a structurally-matching token, rebinds only the buffers (which is cheap)
//! instead of rebuilding the kernels. It is correctness-safe by construction:
//! the cache key carries the operation's structural kernel key (op type,
//! kernel fields, dispatch, workgroup, and every input/output layout/format), so
//! any structural change is a miss (full rebuild). Buffers are *always*
//! re-derived from the current token's inputs/output, so a cached template never
//! carries a stale activation buffer.
//!
//! Disable with `FUSOR_DISABLE_DECODE_PLAN_CACHE` (native only; `var_os` is
//! always `None` on wasm, so the cache is always on for the web build).

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHasher};

use crate::mir::inputs::MirValue;
use crate::mir::kernel_backend::{
    DirectKernel, DirectKernelTemplate, KernelCacheKey, KernelVariantKey,
};
use crate::mir::operation::Operation;
use crate::mir::workgroup_shape::WorkgroupShape;
use crate::tensor::TensorData;

struct DecodePlanCacheKernelVariant;

pub(crate) fn structural_kernel_key(
    operation: &dyn Operation,
    inputs: &[MirValue],
    workgroup: &WorkgroupShape,
) -> KernelCacheKey {
    let dispatch_size = operation.dispatch_size(workgroup, inputs);
    operation.kernel_cache_key_with_dispatch(
        KernelVariantKey::of::<DecodePlanCacheKernelVariant>(),
        Some(workgroup),
        dispatch_size,
        inputs,
    )
}

/// Where one of a kernel's bound buffers comes from on each decode token.
enum BufSource {
    /// The operation's output buffer (freshly allocated every token).
    Output,
    /// The buffer of `inputs[i]` (a graph tensor or a quantized matrix).
    Input(usize),
    /// A buffer that is neither an input nor the output — a model weight or a
    /// scratch buffer allocated during the build. Stable across tokens, so we
    /// hold the `Arc` and reuse it. (Reuse is safe: submits execute in queue
    /// order, so a scratch buffer is never read by token N while token N+1
    /// writes it.)
    Const(Arc<wgpu::Buffer>),
}

/// A memoized build result for a single operation: the kernel templates plus,
/// for each kernel, where to source its buffers from on a replay.
struct CachedOpPlan {
    kernel_key: KernelCacheKey,
    alias_fp: u64,
    kernels: Vec<(DirectKernelTemplate, Vec<BufSource>)>,
}

impl CachedOpPlan {
    fn record(
        kernel_key: KernelCacheKey,
        alias_fp: u64,
        built: &[DirectKernel],
        inputs: &[MirValue],
        output: &TensorData,
    ) -> Self {
        let output_buf = output.buffer();
        let kernels = built
            .iter()
            .map(|kernel| {
                let sources = kernel
                    .binding_buffers()
                    .iter()
                    .map(|buffer| classify_buffer(buffer, inputs, output_buf))
                    .collect();
                (kernel.to_template(), sources)
            })
            .collect();
        Self {
            kernel_key,
            alias_fp,
            kernels,
        }
    }

    fn matches(&self, kernel_key: KernelCacheKey, alias_fp: u64) -> bool {
        self.kernel_key == kernel_key && self.alias_fp == alias_fp
    }

    fn rebind(&self, inputs: &[MirValue], output: &TensorData) -> Vec<DirectKernel> {
        self.kernels
            .iter()
            .map(|(template, sources)| {
                let new_buffers = sources
                    .iter()
                    .map(|source| match source {
                        BufSource::Output => output.buffer().clone(),
                        BufSource::Input(i) => mir_buffer(&inputs[*i])
                            .expect("cached Input source must resolve to a buffer")
                            .clone(),
                        BufSource::Const(buffer) => buffer.clone(),
                    })
                    .collect::<Vec<_>>();
                template.bind_buffers(&new_buffers)
            })
            .collect()
    }
}

fn mir_buffer(value: &MirValue) -> Option<&Arc<wgpu::Buffer>> {
    match value {
        MirValue::Tensor(tensor) => Some(tensor.buffer()),
        MirValue::QMatrix(matrix) => Some(matrix.buffer()),
        MirValue::Integer(_) | MirValue::Float(_) => None,
    }
}

fn classify_buffer(
    buffer: &Arc<wgpu::Buffer>,
    inputs: &[MirValue],
    output_buf: &Arc<wgpu::Buffer>,
) -> BufSource {
    if Arc::ptr_eq(buffer, output_buf) {
        return BufSource::Output;
    }
    for (i, input) in inputs.iter().enumerate() {
        if let Some(input_buf) = mir_buffer(input)
            && Arc::ptr_eq(buffer, input_buf)
        {
            return BufSource::Input(i);
        }
    }
    BufSource::Const(buffer.clone())
}

fn fingerprint_aliases(inputs: &[MirValue], output: &TensorData) -> u64 {
    let mut hasher = FxHasher::default();
    1u8.hash(&mut hasher);
    inputs.len().hash(&mut hasher);
    let mut seen = Vec::new();
    for input in inputs {
        let class = mir_buffer(input).map(|buffer| alias_class(&mut seen, buffer));
        class.hash(&mut hasher);
    }
    alias_class(&mut seen, output.buffer()).hash(&mut hasher);
    hasher.finish()
}

fn alias_class<'a>(seen: &mut Vec<&'a Arc<wgpu::Buffer>>, buffer: &'a Arc<wgpu::Buffer>) -> usize {
    if let Some(index) = seen
        .iter()
        .position(|candidate| Arc::ptr_eq(*candidate, buffer))
    {
        index
    } else {
        let index = seen.len();
        seen.push(buffer);
        index
    }
}

/// The per-resolve view of the cache: the slots for the current graph size,
/// taken out of the shared map so the resolve loop can read/write them without
/// locking on every op. Returned to the cache via [`DecodePlanCache::put`].
pub(crate) struct OpPlanSlots {
    op_count: usize,
    slots: Vec<Option<CachedOpPlan>>,
    hits: u32,
    misses: u32,
}

impl OpPlanSlots {
    /// Return the kernels for operation `index`, reusing the cached build if the
    /// op is structurally unchanged, otherwise invoking `build` and caching the
    /// result. `kernel_key` must uniquely describe the generated kernel IR for
    /// the current operation, including every input and output layout/format.
    pub(crate) fn resolve_op(
        &mut self,
        index: usize,
        kernel_key: KernelCacheKey,
        inputs: &[MirValue],
        output: &TensorData,
        build: impl FnOnce() -> Vec<DirectKernel>,
    ) -> Vec<DirectKernel> {
        let alias_fp = fingerprint_aliases(inputs, output);

        if let Some(Some(plan)) = self.slots.get(index)
            && plan.matches(kernel_key, alias_fp)
        {
            self.hits += 1;
            return plan.rebind(inputs, output);
        }

        self.misses += 1;
        let built = build();
        if index < self.slots.len() {
            self.slots[index] = Some(CachedOpPlan::record(
                kernel_key, alias_fp, &built, inputs, output,
            ));
        }
        built
    }
}

/// Persistent, device-wide cache of decode plans, keyed by graph size so that
/// distinct graphs (prefill / decode / sampler) never evict one another.
pub(crate) struct DecodePlanCache {
    enabled: bool,
    by_op_count: Mutex<FxHashMap<usize, Vec<Option<CachedOpPlan>>>>,
}

impl std::fmt::Debug for DecodePlanCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodePlanCache")
            .field("enabled", &self.enabled)
            .finish()
    }
}

impl DecodePlanCache {
    pub(crate) fn new() -> Self {
        Self {
            enabled: std::env::var_os("FUSOR_DISABLE_DECODE_PLAN_CACHE").is_none(),
            by_op_count: Mutex::new(FxHashMap::default()),
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Take the slot vector for a graph of `op_count` operations, sized exactly
    /// to `op_count` (padding with empty slots / truncating as needed).
    pub(crate) fn take(&self, op_count: usize) -> OpPlanSlots {
        let mut slots = self
            .by_op_count
            .lock()
            .remove(&op_count)
            .unwrap_or_default();
        slots.resize_with(op_count, || None);
        OpPlanSlots {
            op_count,
            slots,
            hits: 0,
            misses: 0,
        }
    }

    /// Return slots taken via [`take`](Self::take) to the shared cache.
    pub(crate) fn put(&self, slots: OpPlanSlots) {
        // Mirror the resolve host-trace gate (always on for wasm, env-gated on
        // native) so the cache hit rate shows up alongside `resolve_host_profile`.
        let trace =
            cfg!(target_arch = "wasm32") || std::env::var_os("FUSOR_TRACE_RESOLVE_HOST").is_some();
        if trace && (slots.hits != 0 || slots.misses != 0) {
            let msg = format!(
                "decode_plan_cache op_count={} hit={} miss={}",
                slots.op_count, slots.hits, slots.misses
            );
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&msg.into());
            #[cfg(not(target_arch = "wasm32"))]
            eprintln!("{msg}");
        }
        self.by_op_count.lock().insert(slots.op_count, slots.slots);
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

    // Exercises the `Sequence` kernel arm (scratch + reduce + write passes) with
    // its scratch buffer reused across tokens: a large softmax takes the split
    // path. `seed` makes the input (and output) input-dependent.
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
    fn decode_plan_cache_rebind_matches_fresh_build() {
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

            // --- Sequence arm (split softmax with a reused scratch buffer) ---
            let miss_v = run_softmax(&device, 0.0).await;
            let hit_v = run_softmax(&device, 0.0).await;
            assert_eq!(
                miss_v, hit_v,
                "softmax: rebind must byte-match the cold build"
            );
            let hit_u = run_softmax(&device, 0.7).await;
            let golden_u = run_softmax(&golden, 0.7).await;
            assert_close(&hit_u, &golden_u, "softmax: rebind with new buffers");
            assert!(hit_u != hit_v, "softmax: different inputs must differ");
        });
    }

    #[test]
    fn decode_plan_cache_distinguishes_same_shape_generic_ops() {
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
    fn decode_plan_cache_distinguishes_same_shape_slice_assign_ranges() {
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
