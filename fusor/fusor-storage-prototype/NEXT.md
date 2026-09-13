# Symbolic indexing, shared collectives, and owned execution

This iteration builds three of the architectural follow-ups. Logical stages
still describe arithmetic and logical reads. Index algebra proves ownership,
the storage planner assigns addresses, a shared collective recipe emits the
reduction algorithm, and an executable owns the resources needed for replay.
The normal Session compiler has not been replaced by the prototype planner.

## Symbolic indexing

`index.rs` represents nonnegative integer maps with constants, variables,
weighted sums, quotients, and remainders. Normalization composes views and
cancels identities such as `n * (x / n) + x % n = x`. Bounds eliminate redundant
divisions and remainders. Analysis and WGSL addressing use the same expression
representation; the eager numerical reference remains independent.

`analysis.rs` records one access recipe per logical read, with reduction-loop
variables and bounds. To prove ownership it substitutes
`output = group * output_share + local`, divides the mapped source index by
`source_share`, and requires the normalized owner to equal `group`. An unproved
map is rejected conservatively. Normal planning has no finite fallback and
allocates no per-element read vectors.

Representative planning measurements on Apple M2 Max:

| Input | Softmax planning | Access recipes |
| --- | ---: | ---: |
| 16 × 128 | 0.137 ms | 6 |
| 128 × 1024 | 0.280 ms | 6 |
| 2048 × 4096 | 0.340 ms | 6 |

The previous cached finite analysis took about 73 ms at 128 × 1024, versus
roughly 0.2–0.3 ms now. This historical comparison establishes the scale of the
host improvement, not a matched compiler microbenchmark. Search still depends
on graph size and candidate partitions. [Scaling run](results/symbolic-scaling.txt).

`--audit-indices` independently enumerates the small graph's numerical reads,
compares them with symbolic recipes, and verifies that every accepted ownership
proof is also legal under enumeration. Ninety plans passed across five shapes
and default, materialized, and legacy policies. The audit checks soundness of
accepted proofs, not completeness of the solver. [Audit run](results/symbolic-audit.txt).

## One collective algorithm, two adapters

`fusor-gpu/src/reduction.rs` exposes a `CollectivePlan` and a small emitter
interface. The plan supplies both the scratch requirement and algorithm:
subgroup collective, leader partial stores, synchronization, and partial merge.
Naga and WGSL adapters provide values and addresses. The existing Naga path
uses the same instruction and barrier order as before this extraction.

On this GPU the subgroup width is fixed at 32. A 64-lane reduction needs two
shared partials: **8 bytes instead of 256 bytes** for the portable tree. The
storage planner sees this smaller requirement before deciding which regions
fit. A single-subgroup block would need no shared partials. The leading barrier
protects reuse of scratch, including the previous iteration of a reduction loop.

Subgroups are selected only when the device reports a fixed width dividing the
block. `--no-subgroups`, legacy mode, and plans created without a device retain
the portable tree. Native WGSL uses the pinned Naga parser's subgroup intrinsics
and builtins; that parser does not accept an `enable subgroups` directive. Browser
execution has not been validated.

Matched warm GPU measurements, 128 × 1024, 16 KiB shared cap, microseconds/step:

| Graph | Portable tree | Native collective | Tree / native |
| --- | ---: | ---: | ---: |
| Reshape/reduction | 4.296 | 3.592 | 1.20× |
| Softmax | 15.010 | 13.570 | 1.11× |
| Fanout | 9.631 | 9.060 | 1.06× |
| Transpose/reduction | 9.796 | 7.966 | 1.23× |
| Global reduction | 8.373 | 7.131 | 1.17× |
| Reuse, no collective | 8.594 | 8.528 | 1.01× |

Both prototype variants use symbolic indexing, scalar forwarding, and the same
cost parameters. Only collective selection and its scratch requirement differ;
the planner may therefore choose a different region under a tight memory cap.
The reuse case emits the same shader in both variants and serves as a timing
control. [Larger run](results/next-bench-large.txt).

The small global reduction remains slower than existing Fusor: 4.922 versus
4.062 microseconds, about 21% longer. Its portable version takes 5.588
microseconds. GPU clock variation is visible in the small-run ranges; these
medians should not be read as fixed latency guarantees. [Small run](results/next-bench-small.txt).

## Owned executable

The unsafe raw-record capture has been replaced with an opt-in owned API:

```rust,ignore
let executable = target.launcher().capture_executable(|| {
    device.session().resolve(&roots)
})?;
// Capture executes that initial resolve normally.
executable.write(input_buffer, 0, input_bytes)?;
executable.submit();
// Or executable.encode(&mut pass) for a caller-owned batch.
```

`GpuExecutable` retains every bound pool lease alongside pipelines, bind groups,
and dispatch grids. A WGPU bind group alone keeps the GPU allocation alive but
does not prevent the Fusor pool from lending that allocation to another graph.
Holding the original `Buf` leases closes that ownership gap. Replay through
`encode` performs no graph resolution, allocation, cache lookup, or rebinding.
`submit` creates a command encoder/pass around one replay.

`inputs()` exposes non-uniform bindings that are read but never written by the
program. `write` accepts only those buffers and checks usage, alignment, and
bounds. Shapes, uniforms, and buffer identities remain fixed. Changing shapes
or identities requires another executable. Capture is synchronous, excludes
other threads, clears state on errors/unwinding, and rejects nested, empty,
already-resolved, copy-containing, or unretained-command captures.

The benchmark forces same-size allocator churn, overwrites the fresh buffers,
then checks three changing inputs without rebuilding the graph. It also checks
rejected writes and failed/empty capture cleanup. This passed on all five cases
that the working-tree Fusor compiler compiles. That compiler's fanout case
reports a 32-versus-256-lane group error. The isolated staged build, excluding
other pending compiler edits, passes all six captures including fanout and both
outputs. [Isolated replay checks](results/next-clean-replay.txt).

This is an opt-in fixed dispatch program, not automatic session caching. Keeping
leases can increase the allocator's live working set during capture. Independent
overlapping executions need independent instances, and mutable state, copies,
dynamic rebinding, and full training replay need further design.

## Validation and reproduction

The GPU matrix passed **360 independent reference comparisons** across native
and portable execution, tail shapes, three packing policies, disabled fusion,
and disabled forwarding. Arenas are poisoned between inputs; allocation and
WGSL validation also run. [Matrix results](results/next-correctness.txt).

The existing reduction conformance suite passed 52 CPU/GPU results after the
Naga adapter extraction. The GPU crate also checks with default features
disabled. Isolated integration checks additionally validate the staged code
against the committed allocator, excluding unrelated working-tree changes.
[Conformance results](results/next-clean-reductions.txt).

The recorded benchmarks use the current working branch on Apple M2 Max, native
Metal, on 2026-09-13. Each of four variants gets five warmup batches and twelve
measurement rounds, rotating order so each occupies every position equally.
Timestamp queries surround the compute pass; planning, allocation, uploads,
host encoding, and readback are excluded. Host timings are reported separately.
Legacy policies now also use the symbolic index emitter, so they are a policy
control, not a byte-for-byte reconstruction of the original prototype.

From `fusor/`:

```sh
cargo run --offline --release -p fusor-storage-prototype -- --all --plan-only --audit-indices --rows 7 --cols 130
cargo run --offline --release -p fusor-storage-prototype -- --all --no-subgroups
cargo run --offline --release -p fusor-storage-prototype -- --bench --all --rows 128 --cols 1024 --shared-bytes 16384 --iterations 200
cargo run --offline --release -p fusor-conformance -- reductions
```

Remaining architectural limits: static f32 graphs, at most twelve materialized
stages, contiguous ownership partitions in one topological order, heuristic
coloring/packing and cost estimates, and no automatic register tiling or general
rematerialization. These results do not establish optimal fusion or end-to-end
training speed.
