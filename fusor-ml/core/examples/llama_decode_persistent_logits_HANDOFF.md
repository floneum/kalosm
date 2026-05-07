# Llama Decode Handoff: Fused Logits + Persistent Decode

## Goal

Get Llama 3.1 8B chat decode from the current warmed fast path of roughly
`38-40 ms/token` down toward the `~20 ms/token` target. Isolated qmat kernels
are no longer the obvious gap; the remaining large wins are:

1. Fuse final logits/top-k/sampling so decode does not materialize full vocab
   logits and then launch a second sampler pipeline.
2. Add a persistent/replay decode path so each token does not rebuild/lower/
   prepare the same 300+ dispatch graph.

Do not add WGSL. New kernels should be raw Naga through
`fusor-ml/tile-ir`.

## Current Measurements

Representative low-overhead full decode command:

```bash
KALOSM_TRACE_DECODE_TIMING=1 \
KALOSM_LLAMA_GPU_SAMPLE_TOKEN=1 \
KALOSM_LLAMA_GPU_SAMPLE_TOP_K=1 \
KALOSM_PROFILE_LLAMA_SOURCE=llama-3.1-8b-chat \
KALOSM_PROFILE_LLAMA_WARMUP=2 \
KALOSM_PROFILE_LLAMA_TOKENS=6 \
target/release/examples/profile_forward
```

Warmed fast decode after the current patch stack:

```text
forward_graph_build: ~0.7-0.9 ms
forward_resolve:     ~12-13 ms
sample/download:     ~24-25 ms
total:               ~38-40 ms/token
kernels:             307 forward kernels
```

GPU timestamp profile on warmed fast decode:

```text
q_mat total:         ~16.6-17.1 ms across 145 kernels
q_mat_swiglu:        ~7.9-8.1 ms across 32 kernels
rms_norm:            ~0.76 ms across 65 kernels
flash_attention:     ~0.55 ms across 32 kernels
nary/select/add:     sub-ms total
```

Host profile on warmed fast decode:

```text
resolve total:       ~12-14 ms
build_execution_graph ~0.7-0.8 ms
optimize:            ~2.4 ms after guard patch
build_kernel:        ~3.4-3.7 ms
prepare_dispatch:    ~1.3-1.9 ms
submit:              ~2.2-3.0 ms
```

The first fast token after fallback is noisy and can show 100+ ms due pipeline
warmup/Metal compilation. Ignore it for steady-state.

## What Has Already Been Tried

The token-10 cliff was addressed:

- KV cache nodes now support batched cached detaching/rebasing.
- Chat-style unbounded decode now reserves a default 256 decode tokens via
  `KALOSM_LLAMA_UNBOUNDED_DECODE_RESERVE`.
- This prevents the cache graph and backing storage from growing every few
  generated tokens.

SwiGLU/qmat work:

- Added raw-Naga tile `StoreSwiGlu` so subgroup reductions stay uniform, but
  `gate * sigmoid(gate) * up` only computes under the store-lane mask.
- Focused `q_mat_swiglu_f32_1x1x4096_Q4k_28672x4096` improved from roughly
  `470 us` mean to roughly `390 us` mean.
- `8x2` SwiGLU tile is available and looked slightly better in the focused
  bench, but full model did not clearly improve versus `4x2`.
- Local ggml Metal same-shape Q4K GEMV microbench was slower than Fusor on this
  machine: ggml around `838 us`, Fusor around `382 us` for non-SwiGLU
  `1x4096 @ Q4K(28672x4096)`.

Conclusion: keep qmat improvements, but do not expect isolated qmat tuning to
find the missing 18-20 ms by itself.

## Key Code Pointers

Forward/model path:

```text
models/kalosm-llama/src/raw/mod.rs
models/kalosm-llama/src/raw/attention_layer.rs
models/kalosm-llama/src/model.rs
models/kalosm-llama/src/raw/cache.rs
```

Final logits are currently built here:

```text
models/kalosm-llama/src/raw/mod.rs
let x = self.norm.forward_generic(&layer_in);
let x = x.i((.., seq_len - 1, ..));
let x_f32 = x.cast::<f32>();
let result_f32 = x_f32.q_mat_mul(&self.output);
```

GPU sampling currently starts only after `device.resolve_batch(&[logits_key])`:

```text
models/kalosm-llama/src/model.rs
forward_sample_token(...)
```

Sampler kernels are in:

```text
fusor-ml/core/src/top_k.rs
```

Important: those existing sampler kernels are WGSL. New fused work should be
raw Naga through `fusor-tile-ir`, not more WGSL.

Graph resolver / dispatch path:

```text
fusor-ml/core/src/compute_graph/mod.rs
fusor-ml/core/src/compute_graph/resolve.rs
fusor-ml/core/src/mir/direct_kernel.rs
```

Raw Naga tile source:

```text
fusor-ml/tile-ir/src/tile.rs
fusor-ml/tile-ir/src/lower/tile_program.rs
fusor-ml/tile-ir/src/lower/quantized.rs
```

## Workstream A: Fused Final Logits + Top-K/Sample

Current expensive shape:

```text
hidden: 1 x 4096
lm_head: Q6K, 128256 x 4096
logits: 1 x 128256
```

Today the path is:

1. Resolve full forward graph, including final LM-head qmat to a full logits
   tensor.
2. Submit/wait.
3. Run GPU sampler pipeline over full logits:
   `adjust logits -> chunk top-k -> merge top-k -> mirostat sample -> copy u32`.
4. Map one token id back to host.

That second phase is about `21-25 ms` in warmed low-overhead runs. It is the
largest single visible post-forward cost.

Target design:

- Add a decode-only fused output op that consumes the final hidden vector and
  the LM-head `QMatrix`.
- It should compute only top-k candidates, not materialize all full logits.
- It should apply generation processors while producing candidates:
  temperature and repetition penalty are enough for the current GPU Mirostat2
  path.
- Then merge candidates and run Mirostat2 sampling in the same command encoder,
  returning a single `u32` token id.

Likely implementation shape:

- Start with a specialized raw-Naga qmat-topk op for `m=1` final logits:
  each workgroup/subgroup owns one or more vocab columns, computes dot, applies
  processors, and keeps local top candidates.
- Emit compact per-chunk candidate buffers: ids + values.
- Reuse or rewrite merge/sample as raw Naga. Avoid adding new WGSL.
- Wire through a new `Model::forward_sample_token_fused_logits` path that skips
  `result_f32 = x_f32.q_mat_mul(&self.output)` as a graph node when GPU sampling
  is active.

Expected win:

- Avoid full logits allocation/write/read by sampler.
- Avoid one full `resolve_batch` boundary before sampler.
- Replace several sampler kernels and a full-logits processor pass with a
  qmat-topk path.
- This is the most plausible path to reclaim most of the current `21-25 ms`
  sample/download bucket.

Correctness constraints:

- Must preserve token ids and logits ordering for top-k candidates.
- Respect repetition penalty over the fixed previous-token window.
- Keep Mirostat2 state update identical or very close to existing
  `GpuMirostat2Sampler`.
- Return one `u32`, not a downloaded logits vector.

## Workstream B: Persistent Decode / Replay Path

Current fast decode still rebuilds the same logical graph each token:

```text
~969 queued ops
307 kernels
build_execution_graph + optimize + lower + build_kernel + prepare_dispatch every token
```

The warmed host portion is still roughly `12 ms`, even after small cache and
optimizer improvements.

Target design:

- Build a decode plan once after the first fast-decode token has stable shapes.
- Cache the ordered direct dispatch plan keyed by:
  model/layer weights, sequence decode shape, KV capacity/layout, sampler mode,
  and relevant tensor layouts.
- On subsequent tokens, update only dynamic inputs:
  token id, position/index, KV write offsets, previous-token sampler buffer, RNG
  params.
- Replay/encode the prepared dispatch list without repeating execution graph
  construction, rewrite optimization, raw-Naga lowering, pipeline lookup, and
  bind-group preparation.

Possible staged approach:

1. Add instrumentation/cache identity only: prove decode shapes and dispatch
   names are stable across tokens after reservation.
2. Cache `PreparedDirectDispatch`-like records, but do not cache output buffers
   yet. This may need rebinding if output buffers rotate through the allocator.
3. Move transient decode buffers into a stable per-session scratch arena so bind
   groups can be reused safely.
4. Add a `resolve_decode_plan(...)` or `Device::replay_decode(...)` API used by
   `Model::forward_sample_token`.

Expected win:

- Remove most of `build_execution_graph`, `optimize`, `build_kernel`, and
  `prepare_dispatch`.
- Best case saves `6-9 ms/token` host-side before any GPU work starts.
- Combined with fused logits/top-k, this is the realistic route to `~20 ms`.

Risks:

- KV cache mutation means cached buffer identities and slice offsets must be
  stable. The unbounded reserve and detach work was added specifically to make
  this more feasible.
- Current allocator can reuse buffers opportunistically. A replay path probably
  wants an explicit decode scratch arena or stable per-op output slots.
- Any replay cache must invalidate on shape/capacity changes, model switch,
  quantization/layout changes, and sampler mode changes.

## Validation Commands

Build:

```bash
cargo check -p kalosm-llama --example profile_forward
cargo build -p kalosm-llama --example profile_forward --profile release
```

Focused qmat:

```bash
FUSOR_QMAT_BENCH_SWIGLU=1 \
FUSOR_QMAT_BENCH_WARMUP_BATCHES=2 \
FUSOR_QMAT_BENCH_MEASURED_BATCHES=20 \
FUSOR_QMAT_BENCH_DISPATCHES_PER_BATCH=16 \
target/release/examples/bench_llama_qmat_bottleneck
```

Full decode:

```bash
KALOSM_TRACE_DECODE_TIMING=1 \
KALOSM_LLAMA_GPU_SAMPLE_TOKEN=1 \
KALOSM_LLAMA_GPU_SAMPLE_TOP_K=1 \
KALOSM_PROFILE_LLAMA_SOURCE=llama-3.1-8b-chat \
KALOSM_PROFILE_LLAMA_WARMUP=2 \
KALOSM_PROFILE_LLAMA_TOKENS=6 \
target/release/examples/profile_forward
```

Host profile:

```bash
FUSOR_TRACE_RESOLVE_HOST=1 \
FUSOR_TRACE_RESOLVE_HOST_CATEGORIES=1 \
KALOSM_TRACE_DECODE_TIMING=1 \
KALOSM_LLAMA_GPU_SAMPLE_TOKEN=1 \
KALOSM_LLAMA_GPU_SAMPLE_TOP_K=1 \
KALOSM_PROFILE_LLAMA_SOURCE=llama-3.1-8b-chat \
KALOSM_PROFILE_LLAMA_WARMUP=1 \
KALOSM_PROFILE_LLAMA_TOKENS=3 \
target/release/examples/profile_forward
```

GPU kernel profile:

```bash
FUSOR_TRACE_GPU_KERNELS=1 \
KALOSM_TRACE_DECODE_TIMING=1 \
KALOSM_LLAMA_GPU_SAMPLE_TOKEN=1 \
KALOSM_LLAMA_GPU_SAMPLE_TOP_K=1 \
KALOSM_PROFILE_LLAMA_SOURCE=llama-3.1-8b-chat \
KALOSM_PROFILE_LLAMA_WARMUP=1 \
KALOSM_PROFILE_LLAMA_TOKENS=3 \
target/release/examples/profile_forward
```

Conformance for the current SwiGLU raw-Naga changes:

```bash
cargo test -p fusor-conformance q4k_q_mat_mul_swiglu_matches_cpu_reference
```

## Current Diagnostic Knobs

```text
KALOSM_LLAMA_UNBOUNDED_DECODE_RESERVE=256
FUSOR_RESOLVE_SKIP_OPTIMIZE=1
FUSOR_Q4K_SWIGLU_TILE=4x2|8x2|4x1|2x2|2x4|4x4|8x1
```

`FUSOR_RESOLVE_SKIP_OPTIMIZE=1` is diagnostic only. It removes optimizer time
but increases kernel count and did not materially improve end-to-end decode.

`FUSOR_Q4K_SWIGLU_TILE=8x2` helped the focused benchmark but did not clearly
improve the full model versus `4x2`; verify full-model timing before relying on
it as a default.
