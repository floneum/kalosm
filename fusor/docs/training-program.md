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
addresses, keeping bounded batch, row, column and reduction coordinates symbolic
through composed reshape/transpose expressions until shader emission. Tail and
gather guards establish the logical bounds once; helpers retain ownership checks
without repeating the bounds branch. Contiguous reductions use subgroup collectives when available;
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
workgroups advance independently. Compiler tests independently check for
overlapping live allocations. Inputs and retained outputs stay live, and feedback waits
until every old-state read has completed. Scheduling and packing are heuristics;
dispatch count alone is not the optimization objective.

Current admission supports nonempty static shapes, f32/u32/i32 values, scalar
reduction carriers, pointwise operations, views, gathers/scatters and
contractions. Unsupported dtypes, symbolic shapes, invalid feedback, and buffer
budget violations return errors. The program has one storage binding and uses
256 lanes per workgroup. Storage bounds checks remain enabled; statically
bounded loops omit redundant runtime loop counters.

## Browser acceleration

Device creation requests advertised subgroups and probes supported f32 8×8
matrix configurations. `TrainingProgram::acceleration()` reports the selected
instructions; `acceleration_fallback()` explains a matrix fallback. Matrix kernels
require fixed 32-lane subgroups. Ordinary subgroup reductions use the runtime
width, with a shared-memory fallback when subgroup slots are not fully occupied.
The pinned backend fork is documented in [browser-dependencies.md](browser-dependencies.md).

Ordinary Session dispatches bind each physical arena once. Mixed scalar views
load and store through u32 words; packed f16 stores preserve the neighboring half
with compare-exchange. Homogeneous bindings use their native element type.

## Tests and benchmarks

GPU tests compare outputs with host arithmetic and cover simultaneous state
feedback, retained snapshots, changing inputs, tail tiles, reduction boundaries,
indexing, mixed-type storage, and scratch reuse. The transformer test compares
all parameters and Adam moments with Session after 24 steps, checks evaluation
and generation, and trains the compiled model through 400 steps.

Run from `fusor/`:

```sh
cargo test --release -p fusor --features compiler-tests --test program
cargo test --release -p fusor --example train_small
cargo run --release -p fusor-conformance
cargo run --release -p fusor --example train_mlp -- fused
cargo run --release -p fusor --example train_small -- compare 32
```

The training examples support `baseline` and `fused` execution. `train_small`
also supports `single`, `subgroups`, and `portable`. Queued step measurements
include input updates and a readback per window; they exclude initialization,
evaluation, generation, and UI rendering. Recorded protocols and measurements
are in [training-benchmarks.json](training-benchmarks.json) and
[browser-performance.json](browser-performance.json).

Serve the release app from `webgpu-runner/` using Dioxus CLI 0.7:

```sh
dx serve --platform web --release --port 8900
```

Open `http://localhost:8900/#/train`. See the [runner README](../webgpu-runner/README.md)
for configuration, corpus caching, and UI tests.

Browser compiler tests require a separate build with `--features training-checks`.
With Playwright available on `NODE_PATH` and a WebGPU-capable browser in `CHROME`:

```sh
node tests/training.cjs
node tests/general.cjs matmul normalization attention_rope
FUSOR_ROW_MODE=permuted-lanes node tests/general.cjs normalization
FUSOR_ROW_MODE=partial-subgroups node tests/general.cjs normalization
```

The training harness checks portable, subgroup-only, and matrix execution,
including absent capabilities and shader rejection. The normalization harness
can permute local lanes across subgroups or force the occupancy fallback.
Diagnostic exports are absent from production builds.
