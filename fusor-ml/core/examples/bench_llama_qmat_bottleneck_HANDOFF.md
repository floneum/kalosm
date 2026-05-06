# Llama QMat Bottleneck Benchmark Handoff

## Goal

Optimize the Llama 8B decode bottleneck isolated by
`bench_llama_qmat_bottleneck.rs`.

The benchmark targets this profiled kernel:

```text
q_mat_mul_f32_1x1x4096_Q4k_28672x4096
```

This is the fused FFN gate/up projection shape in the Llama 8B forward pass:

```text
A = 1 x 4096
B = 4096 x 28672, Q4K
Y = 1 x 28672
```

In the full decode profile this qmat family was the largest single qmat cost.
Sampler/readback was reduced to a single token id, and throughput did not move
meaningfully, so the remaining bottleneck is forward qmat/GEMV work.

## Benchmark

File:

```text
fusor-ml/core/examples/bench_llama_qmat_bottleneck.rs
```

Build:

```bash
cargo build -p fusor-core --example bench_llama_qmat_bottleneck --profile release
```

Run:

```bash
target/release/examples/bench_llama_qmat_bottleneck
```

If running from the Codex sandbox, this may need to run outside the sandbox so
Metal can create an adapter.

Useful knobs:

```bash
FUSOR_QMAT_BENCH_WARMUP_BATCHES=3
FUSOR_QMAT_BENCH_MEASURED_BATCHES=20
FUSOR_QMAT_BENCH_DISPATCHES_PER_BATCH=16
```

Baseline from the first working run on Apple M2 Max:

```text
mean_dispatch_time_us: 482.040
p50_dispatch_time_us: 501.203
p90_dispatch_time_us: 522.544
min_dispatch_time_us: 435.408
max_dispatch_time_us: 522.794
effective_gflops: 487.264592
packed_weight_bandwidth_gb_s: 137.043167
```

The benchmark batches multiple independent qmat graph nodes into one
`Device::resolve_batch` call and then waits once. This keeps host synchronization
from dominating the measurement. The output tensors are deliberately kept alive
until after `resolve_batch`; otherwise the graph node can be dropped before
resolution.

## Key Code Paths

Fusor qmat operation:

```text
fusor-ml/core/src/quantized/matmul/mod.rs
```

Tile prototype used to generate qgemv kernels:

```text
prototypes/phase-token-prototype/src/tile.rs
```

Q4K/Q6K lowering details:

```text
prototypes/phase-token-prototype/src/lower/quantized.rs
prototypes/phase-token-prototype/src/lower/tile_program.rs
```

QMatrix storage conversion:

```text
fusor-ml/core/src/quantized/mod.rs
```

The qgemv shape selection currently flows through:

```text
qgemv_cols_per_workgroup_for_direct(...)
QMatMulOperation::build_direct_kernel(...)
Program::qgemv_tile(...)
```

For this Q4K shape, the current tile path is selected from the Q4K branch in
`Program::qgemv_tile`.

## Prior Findings

Full Llama 8B decode profile, after moving sampling to GPU:

```text
q_mat dominates the forward pass
largest family: q_mat_mul_f32_1x1x4096_Q4k_28672x4096
next families: Q6K/Q4K FFN down projections
flash attention and norms are much smaller
```

Tried changing the Q4K `rows <= 4096 && cols >= 8192` branch from:

```rust
qgemv_perf::<4, 8, 32, 128>
```

to:

```rust
qgemv_perf::<8, 4, 16, 256>
```

This regressed badly in the existing prototype microbench:

```text
target/release/examples/bench_qmatmul gemv q4k contiguous 1 28672 4096
mean_dispatch_time_us: ~1792.7
p50_dispatch_time_us: ~1521.2
```

That change was reverted. Do not retry it as-is.

Older comparison against ggml/llama.cpp style microbenchmarks showed Fusor was
already competitive on some smaller qmat shapes, but the full decode remains
qmat-bound because this projection appears 32 times per token and the FFN down
projections add more qmat cost.

## What To Optimize

Start with the exact default benchmark shape. A useful improvement should lower
`mean_dispatch_time_us` and `p50_dispatch_time_us` for:

```text
1x4096 @ Q4K(28672x4096)
```

Candidate directions:

1. Tune Q4K qgemv tile shape for large-N, K=4096.
2. Reduce per-column workgroup overhead or improve column packing per workgroup.
3. Improve Q4K dequant/dot path in the lowerer, especially register pressure and repeated scalar work.
4. Look for ways to compute two FFN projections together if the full model path permits it. The benchmark isolates one projection, but the model often uses concatenated gate/up rows.
5. Compare GPU timestamp profiles with `FUSOR_TRACE_GPU_KERNELS=1`, but use low-overhead wall timing for final benchmark numbers because timestamps perturb queue timing.

## Validation

Compile check:

```bash
cargo check -p fusor-core --example bench_llama_qmat_bottleneck
```

Release benchmark:

```bash
cargo build -p fusor-core --example bench_llama_qmat_bottleneck --profile release
target/release/examples/bench_llama_qmat_bottleneck
```

Optional broader qmat comparison:

```bash
cargo build -p phase-token-prototype --example bench_qmatmul --profile release
target/release/examples/bench_qmatmul gemv q4k contiguous 1 28672 4096
```

Use the focused benchmark for optimization iteration, then confirm the full
Llama decode path still improves:

```bash
cargo build -p kalosm-llama --example profile_forward --profile release
KALOSM_PROFILE_LLAMA_SOURCE=llama-8b \
KALOSM_PROFILE_LLAMA_WARMUP=4 \
KALOSM_PROFILE_LLAMA_TOKENS=8 \
target/release/examples/profile_forward
```
