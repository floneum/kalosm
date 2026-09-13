# Measured performance after optimization

Date: 2026-09-13. Apple M2 Max, native Metal. These are warmed GPU kernel
measurements on six synthetic graphs, not end-to-end training results.

The original prototype was not generally faster than current Fusor. A matched
initial comparison found slower reshape/reduction, softmax, transpose, and global
reduction, with only the reuse case faster. Fewer allocations alone did not make
good kernels. [Initial measurements](results/bench-before.txt).

After the changes below, the prototype is faster on all five comparable larger
cases. The smaller global reduction still loses.

## Current Fusor versus optimized prototype

128 × 1024 inputs, 16 KiB prototype shared-memory cap, 200 steps per batch.
Entries are medians of nine warmed GPU timestamp samples, in microseconds/step.

| Graph | Current Fusor | Optimized prototype | Current / prototype |
| --- | ---: | ---: | ---: |
| Reshape/reduction | 10.876 | 3.887 | 2.80× |
| Softmax | 19.339 | 14.149 | 1.37× |
| Fanout, two outputs | Compiler error | 8.975 | — |
| Transpose/reduction | 12.059 | 9.060 | 1.33× |
| Global reduction | 9.291 | 7.797 | 1.19× |
| Five-stage reuse | 32.308 | 7.090 | 4.56× |

Dispatch counts for the optimized prototype are 1, 1, 1, 1, 2, 1 respectively.
The comparable current compiler counts are 1, 1, error, 2, 2, 5.

Two runs agreed closely: the earlier larger run gave speed ratios of 2.80×,
1.37×, 1.33×, 1.18×, and 4.55×. The fanout baseline error persists at both shapes:
`group member ... lowers at 32 lanes, the group at 256`.

[Larger run](results/bench-large.txt),
[confirmation run, before host-analysis caching](results/bench-large-repeat.txt).

16 × 128 inputs, 2 KiB prototype shared-memory cap, 1000 steps per batch:

| Graph | Current Fusor | Optimized prototype | Current / prototype |
| --- | ---: | ---: | ---: |
| Reshape/reduction | 5.877 | 4.084 | 1.44× |
| Softmax | 5.249 | 4.472 | 1.17× |
| Fanout, two outputs | Compiler error | 4.266 | — |
| Transpose/reduction | 4.794 | 3.719 | 1.29× |
| Global reduction | 4.259 | 5.738 | 0.74× |
| Five-stage reuse | 9.714 | 1.579 | 6.15× |

The small global reduction takes about **35% longer**. Small-case samples also
show more GPU clock variation; raw ranges are retained alongside medians.
[Small run](results/bench-small.txt).

## Changes that produced the result

- **Scalar forwarding:** single-consumer pointwise operations and tiny reductions
  become logical expressions at their consumer. Shared producers and requested
  outputs retain storage. This removes intermediate writes, stage barriers, and
  the transpose ownership boundary in the tested chains. The five-stage reuse
  example becomes one dispatch without adding a layout-specific kernel.
- **Reduction scheduling:** use a serial reduction per output lane for axes up
  to 32 elements. The original threshold was eight. Wider reductions retain the
  generic workgroup tree. The illustrative cost model now charges more for
  serial workgroup-tree iterations, reducing the incentive to fuse a global
  reduction into one poorly parallel workgroup. Its parallel-workgroup estimate
  was increased from 32 to 256. These remain heuristics informed by this device,
  not an occupancy model or autotuner.
- **Counted-loop lowering:** optimized native compilation disables injected loop
  bounding after checking that every generated u32 loop counter terminates
  without overflow. Buffer bounds checks remain enabled. Current Fusor already
  uses trusted module creation with both kinds of checks disabled after its own
  verification. This narrows a backend-compilation difference; it is not a
  benefit attributable to graph coloring.
- **Cached access analysis:** compute logical reads and dependencies once per
  graph, then reuse them for all candidate cuts and ownership partitions. Also
  derive external traffic once per region instead of once per group count.
  The larger softmax planning time fell from about 972 ms to 73 ms; the emitted
  kernels and timing ratios remained essentially unchanged.

The very large gains over the *original prototype* at larger shapes include
removing prototype-specific lowering and scheduling costs. They should not be
presented as architectural speedups over Fusor. Raw measurements of the
intermediate version are retained in
[the run before the final lowering/scheduling update](results/bench-large-before-lowering.txt).

## Benchmark method

`--bench` builds the same operations and shapes through current Fusor, captures
its compiled dispatch records, and builds the two prototype variants on the
**same WGPU device**. All three execute through one runner that sets their actual
pipelines, bindings, and dispatch grids in one compute pass per batch.

Each variant gets five full warmup batches, followed by nine measurement rounds.
Variant order rotates and reverses across rounds. Timestamp queries surround
only the GPU compute pass. Graph construction, planning, compilation, buffer
allocation, uploads, host encoding, and readback are outside that GPU span. Host
encoding/submission/wait medians are reported separately. Metal query resolution
occurs after compute completion, outside both measurements.

This measures repeated execution with resident, identical inputs and reused
outputs. It avoids repeatedly resolving already-computed Fusor tensors, which
would perform no GPU computation. It also deliberately excludes Fusor's session
and graph-replay overhead, so it cannot establish end-to-end application speed.

The capture is behind the opt-in `fusor-gpu/prototype-capture` feature. It leaves
lowering unchanged. The runner retains the baseline graph and avoids further
pool allocations while replaying, then checks the captured outputs against the
eager reference. Captures containing copies or no dispatches are rejected as
unsupported instead of silently benchmarked differently.

## Validation and remaining limits

Both prototype variants check three input patterns against the independent eager
reference before measurement, poisoning and reusing the global arena. Current
Fusor's captured results are checked after replay. All supported comparisons in
the tables passed; the fanout compile error is reported explicitly.

An additional matrix ran all six graphs across nine configurations: 1 × 2,
7 × 130, 31 × 64, 33 × 66, each of the three packing policies at 1 KiB, fusion
disabled, and both forwarding and fusion disabled. **162 GPU reference
comparisons passed**, along with emitted-WGSL validation and allocation checks.
[Correctness matrix](results/optimized-correctness.txt).

The executable builds with capture enabled, and the GPU crate also builds with
default features disabled. The terminal forwarding toggle was exercised.

The remaining problems are substantial: the tiny global-reduction regression,
finite analysis that still takes tens of milliseconds and memory proportional to
enumerated reads, an approximate device-specific cost model, and no full training
workload or browser benchmark. The shared-memory tree could use a tuned subgroup
implementation on native devices. Symbolic access proofs and real schedule
selection are needed before considering production integration.

## Reproduce

From the worktree's `fusor/` directory:

```sh
cargo run --release -p fusor-storage-prototype -- --bench --all --iterations 1000
cargo run --release -p fusor-storage-prototype -- --bench --all --rows 128 --cols 1024 --shared-bytes 16384 --iterations 200
```

`--legacy` restores the original prototype's emission and scheduling policies.
`--no-forwarding` isolates materialization under the new scheduling policy.
The prototype's default terminal view shows the number of forwarded values and
the chosen dispatches, groups, and allocations.
