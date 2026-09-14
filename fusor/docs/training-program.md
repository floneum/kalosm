# Fixed-shape training programs

`fusor::program::TrainingProgram` compiles a forward/backward/update graph into
an owned GPU program with persistent inputs and simultaneous state feedback.
The caller supplies logical tensors and `(state, next_state)` pairs. Layouts,
allocation offsets, workgroup ownership, and barriers stay inside the compiler.

The browser Train page selects the compiled program. `Lm::new` still leaves
`Session` available to benchmark callers; `Lm::compile_training(options)` selects
the program explicitly. `TrainingProgram` can also compile another fixed-shape
workload. Held-out scoring and generation use the selected executor with no
optimizer feedback. Generation reuses an observation program until weights move.

```rust,ignore
let mut program = TrainingProgram::compile(
    &[loss.clone(), next_weights.clone()],
    &[(weights.clone(), next_weights)],
).await?;

program.write(&input, bytemuck::cast_slice(&batch))?;
program.run_async().await?;
let loss_bytes = program.read(&loss).await?;

// Publish an independent GPU snapshot for ordinary graph execution.
program.export(&[weights.clone()])?;
```

Compilation snapshots the current inputs without taking an optimizer step.
`write` updates the private program input, and `run_async` queues a complete
step. At most 32 submissions remain pending before an asynchronous fence;
ordinary steps do not wait. `read` observes a completed output, while `export`
copies all requested values in one submission without a host round trip. An
exported tensor remains a snapshot when subsequent steps change program state.
Export all optimizer state as well as parameters when resuming another executor.

The compiler imports logical definitions directly, canonicalizes graph aliases,
forwards inexpensive scalar operations, and emits shared scalar subexpressions
once. In particular, GELU backward remains an expression graph instead of
expanding into repeated exponential calculations. Roots and aliased feedback
sources receive materialized snapshots; swaps observe the old values of both
states.

Logical index recipes prove whether a consumer reads only its workgroup's
outputs. Failed proofs introduce dispatch boundaries. Small programs whose
largest value has at most 4,096 elements use one workgroup by default; larger
programs use parallel regions. The scheduler forms jobs with independent
ownership domains and packs ready, dependency-independent jobs into one dispatch.
Every parallel contraction distributes complete matrix tiles across a bounded
grid; native and portable grids reflect their actual tile counts. Long
parameter-gradient contractions split the reduction axis into independent partial
products followed by a sum. These partials use the same allocation planner as
ordinary values; matrix operands never bind to an allocation layout.

The same symbolic index representation drives ownership proofs and emitted view
addresses, simplifying composed reshape/transpose expressions before shader
compilation. Contiguous reductions use subgroup collectives when available;
strided reductions distribute adjacent output columns across adjacent lanes.
Embedding-gradient scatters compact matching indices once per output row and
reuse the positions across features, preserving the original f32 addition order.
Scratch writes and barriers explicitly cover every read, including padded tiles,
so redundant driver zero-initialization is disabled. Read-only accumulator seeds
also avoid clearing the output staging tile at each matrix iteration.

Native and browser matrix instructions, and stable indexed reductions, require
fixed 32-lane subgroups. Ordinary subgroup reductions use the runtime subgroup
width and operate independently of matrix support. Portable shared-memory tiling and
scalar indexed reductions remain available. Scheduling depends on work shape,
not the model name or parameter count.

The arena uses largest-first placement with lifetime interference checks.
Within parallel execution, lifetimes extend to region boundaries because
workgroups advance independently. A separate pairwise check rejects overlapping
live allocations. Inputs and retained outputs stay live, and feedback waits
until every old-state read has completed. Scheduling and packing are heuristics;
dispatch count alone is not the optimization objective.

Current admission supports nonempty static shapes, f32/u32/i32 values, scalar
reduction carriers, pointwise operations, views, gathers/scatters and
contractions. Unsupported dtypes, symbolic shapes, invalid feedback, and buffer
budget violations return errors. The program has one storage binding and uses
256 lanes per workgroup. Storage bounds checks remain enabled; statically
bounded loops omit redundant runtime loop counters.

## Measurements

Apple M2 Max, Metal, f32. These are end-to-end queued training timings with
readback at batch boundaries, using identical initial weights and data between
modes. The MLP uses fixed inputs. The transformer draws new batches and updates
Adam's learning-rate input every step. Warmup is excluded from step timings.

| Workload | Executor | Dispatches/step | Warm step time |
| --- | --- | ---: | ---: |
| MLP: 1,280 parameters, batch 4 | Session | 8 | 219–222 µs |
| Same MLP | TrainingProgram | 1 | 38–63 µs |
| Transformer: 240,480 parameters, batch 16, context 64 | Session | 216 | 3.76 ms best; 4.46–8.01 ms other windows |
| Same transformer | Parallel TrainingProgram | 99 | 1.85–1.99 ms |

The alternating transformer comparison uses five 32-step windows per executor
after 17 warmup steps, checks losses after every window, and asserts that the
program's fastest warmed window beats Session's. Best-to-best speedup is 2.03×
in this run. Earlier Session measurements reached 3.28 ms; against that faster
historical baseline, the current 1.85 ms is about 1.8× faster. The previous fused
implementation took 5.15–5.26 ms, making the current version about 2.8× faster.
All window timings are retained; slow reference windows do not inflate the
reported best-to-best speedup. The MLP's best warmed windows improve by 5.8×.

After the alternating run, held-out losses were 2.1768246 (Session) and 2.176824
(program), with identical accuracy 0.35839844. The original 64-step-window
regression also passes. The transformer arena occupies 35,690,896 bytes, versus
87,486,360 bytes for dedicated allocations. The earlier forced single-workgroup
transformer measurement was roughly 98 ms; its matrix workloads benefit from
parallel execution. The small MLP remains one kernel.

Wider matrix tiles and matrix epilogues were measured and discarded: faster
isolated GEMMs or fewer dispatches did not improve the actual training step.

See [the recorded output](training-benchmarks.json). Reproduce from `fusor/`:

```sh
cargo run --release -p fusor --example train_mlp -- baseline
cargo run --release -p fusor --example train_mlp -- fused
cargo run --release -p fusor --example train_small -- compare 32
cargo run --release -p fusor --example train_small -- baseline 32
cargo run --release -p fusor --example train_small -- fused 32
cargo run --release -p fusor --example train_small -- single 32
cargo run --release -p fusor --example train_small -- portable 32
```

## Validation and rollout

The GPU regressions cover changing inputs, independent host SGD and matrix
oracles, tail tiles, required global-reduction boundaries, simultaneous swaps,
strided outputs, long gradient contractions, uniform updates, bounded replay,
independent observation snapshots, packed independent jobs, dense repeated
indices, signed integer scatters, and chained-view gather bounds. The full
transformer test compares every parameter and Adam moment against Session after 24 steps, checks evaluation and
generation, and trains the compiled model through 400 steps.

The MLP benchmark also exposed and now covers an existing grouped-launch bug:
subgroup reductions did not honor the enclosing group's lane count. Maps and
folds now honor the common block size, and group ranges are assigned from the
resulting grids. The existing reduction suite passes all 52 CPU/GPU cases with
no skips.

```sh
cargo test --release -p fusor --test program
cargo test --release -p fusor --lib changing_expression_and_integer_leaf_uniforms
cargo test --release -p fusor-gpu --lib
cargo test --release -p fusor --example train_small compiled_training_matches_reference
cargo run --release -p fusor-conformance -- reductions
cargo check --target wasm32-unknown-unknown --manifest-path webgpu-runner/Cargo.toml
```

The release browser app has also run in local Chrome on Metal: 320 steps,
6.7 ms per training step, held-out loss 1.933, accuracy 41%, and 99 training
dispatches per step. This UI metric excludes scoring and generation overhead;
it is a smoke measurement, not the same benchmark as the native timings above.
Browser WGSL requires nonfinite reduction identities to be constructed at runtime.
Held-out scoring and generation now use compiled observation programs without
feedback, avoiding the older browser Session path that returned zero scores.
Other browsers and GPU vendors still need device testing.

Serve the release app from `webgpu-runner/` with a Dioxus 0.7 CLI:

```sh
dx serve --platform web --release --port 8900
```

Open `http://localhost:8900/#/train`.

### Browser acceleration

Device creation requests subgroups whenever advertised. The browser capability
bridge also reads the actual subgroup widths and f32 8×8 matrix configurations;
fixed programs emit Dawn's matrix dialect directly. Subgroup reductions remain
available without matrices and use the runtime subgroup size. Stable indexed
reductions and matrix tiling currently specialize for fixed 32-wide subgroups.
No f16 conversion is introduced.

`TrainingProgram::acceleration()` reports the selected instructions. An
advertised experimental matrix extension that fails pipeline validation retries
with portable matrix kernels; `acceleration_fallback()` retains the reason.
Missing features, missing configurations, and unknown subgroup widths fall back
before shader creation. Ordinary Session kernels use the same advertised matrix instructions through
the shared Naga WGSL writer. Device initialization probes supported matrix kinds
before admitting them to either executor. The small vendored backend change and
its upstream source are documented in `../vendor/README.md`.

The opt-in browser checks exercise the actual model without the UI's scoring and
sampling overhead:

```sh
# Use Dioxus CLI 0.7.x for this application.
cd fusor/webgpu-runner
dx serve --release --features training-checks --port 8900 --open false --watch false
# In another shell, with Playwright installed / available through NODE_PATH:
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" node tests/training.cjs
```

These exports are omitted unless `training-checks` is enabled. The test launches
its own browser and verifies actual feature requests and emitted shader paths.
It covers portable, subgroup-only, accelerated, no-matrix, no-subgroups,
no-configuration, ranged-width and rejected-matrix cases. Each mode checks
non-square and tail matrices, long-K splitting, retained matrix outputs, row
reductions and ordinary Session subgroup emission against host arithmetic, then
compares training and held-out losses. A shader rejection is injected only in
the isolated test browser.

On the M2 Max / Chrome 152 release benchmark, median warmed training falls from
5.47 ms/step for portable shaders to 4.92 ms with subgroup collectives and
4.28 ms with f32 matrix instructions as well. Native remains 1.88 ms/step.
These are five windows of 32 steps after 17 warmup steps; initialization,
held-out scoring, text generation and rendering are excluded. Full measurements
and feature audits are in `training-benchmarks.json`.


### Browser performance follow-up

The browser compiler now removes redundant leading-coordinate modulo operations
inside matrix tile bounds guards. Inner coordinates of multidimensional indices
still wrap at their own extents, and tail loads/stores remain guarded. On the
M2 Max, this reduces warmed full-transformer training from about 4.3 to 3.2 ms per
step without changing precision, arithmetic order, the model, or its 99 dispatches.
A contemporaneous alternating native comparison remains around 2.0 ms per step.

GPU timestamps measured 4.26 ms inside the original training pass versus roughly
4.30 ms wall time. Instrumented WebGPU API calls consumed about 0.084 ms per step;
that API measurement excludes other WASM host work. The principal training gain
comes from shader indexing, not fewer host calls. The 1 ms browser target has not
been reached.

General browser matrix benchmarks also use the shared browser matrix serializer:
256-square GEMM improves from 0.63 to about 0.10 ms, batched GEMM from 0.11 to 0.033
ms, and the attention cases from 0.47 to about 0.12 ms. Layer normalization and
small convolution remain hotspots, around 0.49 and 0.47 ms respectively. The
recorded measurements and limitations are in [browser-performance.json](browser-performance.json).

The checks include an aligned ordinary Session matrix before changing shapes
promote its family to a generic symbolic plan, then non-square/tail matrices and
long-K reductions. Browser normalization and dense-matrix conformance can also
be run with `node tests/general.cjs` using the same opt-in build and environment
as `tests/training.cjs`. Arbitrary conformance filters can be supplied as arguments.
A broader sweep found an existing `matmul::q_mat_mul_rank1` browser failure
(expected -0.625, got 0 at sampled shape [8,1]); it reproduces with matrix support
masked off and remains an outstanding issue. This is not a claim that the entire
browser conformance suite passes.

Larger K tiles, per-job pipeline specialization, a static arena extent, an
additional subgroup-index clamp, composite-expression caching, and a speed-first
WASM release profile were measured and discarded. The profile change more than
doubled the WASM download (11.8 to 27.8 MB) without a useful steady-state gain.
