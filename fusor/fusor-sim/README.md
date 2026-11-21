# fusor-sim

`fusor-sim` is a standalone native tool crate in the Fusor workspace. It lowers real selected GPU launch nodes to `fusor_ir::ir::kernel::KernelIr` through `fusor_gpu::lower_kernel_ir`, then interprets that KernelIr without requesting an adapter or running the CPU backend. It is not a production runtime dependency.

## Boundary and supported subset

The simulator validates behavior at the `KernelIr` boundary. Its independent host reference checks the generated recipe corpus end-to-end, but does not provide general coverage of arbitrary logical-to-launch rewrite correctness. It does **not** validate Naga/WGSL emission, vendor compiler behavior, drivers, GPU timing, session caches, or the timed autotune race.

The current baseline supports f32 arithmetic and scalar map, generic fold/reduction and generic matmul lowering; u32/i32/bool scalar values; lane/workgroup builtins; scalar/vector construction, component selection and dot; locals, conditionals, counted and uncounted loops, break/return; scalar storage and workgroup loads/stores; atomic add; barriers; bounds and initialization checks; and unordered storage/workgroup conflict checks. Subgroups, cooperative matrices, f16/bf16, quantized loads, cooperative stores, bounded `FillTile` (including 2D bounds), repeated or multiple expression reductions in one statement, expression reductions in loop accumulator updates, vector/typed storage, duplicate memory-ID alias bindings, and unrecognized IR return explicit `Unsupported` errors. Reductions must use a supported workgroup form. Simulation is bounded by invocation, memory, and total-step limits.

Lane scheduling uses a locally defined SplitMix64 PRNG and records every typed statement-boundary decision as an explicit lane choice, the runnable lane list, and its digest. JSON traces are versioned and reject program/input changes, runnable-set changes, impossible lanes, truncation, and trailing choices. A seed is only a trace generator; replay consumes the recorded choices.

## Shuttle mechanics

The design borrows only replay mechanics, not Shuttle's concurrent-task execution model or Fusor's GPU schedule-domain semantics. Shuttle's scheduler records a seed and typed `Task`/`Random` decisions in [`shuttle-engine/src/scheduler/mod.rs`](https://github.com/awslabs/shuttle/blob/main/shuttle-engine/src/scheduler/mod.rs); [`shuttle-schedulers/src/replay.rs`](https://github.com/awslabs/shuttle/blob/main/shuttle-schedulers/src/replay.rs) enforces that a recorded task is runnable at the next choice. `fusor-sim` adapts that strict runnable-set validation to lane continuations and additionally records a runnable-set digest/list.

## Commands

```sh
cargo run --manifest-path fusor/Cargo.toml -p fusor-sim -- campaign --seed 1 --iterations 12
cargo run --manifest-path fusor/Cargo.toml -p fusor-sim -- replay target/fusor-sim-first-failure.json
cargo run --manifest-path fusor/Cargo.toml -p fusor-sim -- minimize target/fusor-sim-first-failure.json
cargo test --manifest-path fusor/Cargo.toml -p fusor-sim
```

The campaign reports cases, distinct recipe shapes, enumerated candidates, executed plans, unsupported plans, execution/candidate errors, dispatches, and simulated steps. It includes non-contiguous transpose views, dependent matmul-then-row-reduction recipes, and map/reduction tails around 32- and 64-lane boundaries. Executed plans are capped per case separately from the candidate queue. The first failure is saved as versioned JSON (including recipe, inputs, caps profile, candidate selections, per-dispatch schedule traces, and all execution limits) and the command exits nonzero with a replay command. `--max-steps` overrides the default total-step limit for campaigns; the selected limit is recorded in each failure artifact and used by replay/minimize. Candidate replan diagnostics use `LocalSearch`'s compiler-tests exhaustive `fair_points`/`apply_variant` path so admitted replan failures are not silently filtered. Candidate failures from arbitrary invalid selections are not generated.
