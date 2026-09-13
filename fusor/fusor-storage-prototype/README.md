# Fusor storage/fusion prototype — THROWAWAY

**Question:** Can pointwise, reduction, and reshape operations remain independent
of physical layouts, while a region planner decides fusion using workgroup
ownership and a storage planner colors and packs their live values?

Run from the worktree's `fusor/` directory:

```sh
cargo run --release -p fusor-storage-prototype
```

This opens a small interactive terminal view. Press a key and Enter to switch
cases, packing policy, fusion, or the workgroup-memory budget. Allocation details
show interference lifetimes, byte offsets, and the peak-live lower bound. GPU
execution checks three different inputs against an independent eager reference.
`s` toggles scalar forwarding. Optimized emission and scheduling are now the
default; `--legacy` reproduces the original prototype's policies.

For the warm GPU performance comparison:

```sh
cargo run --release -p fusor-storage-prototype -- --bench --all --rows 128 --cols 1024 --shared-bytes 16384 --iterations 200
```

This compares current Fusor, the original prototype, and the optimized prototype
using the same device and timestamped command-batch runner. See
[PERFORMANCE.md](PERFORMANCE.md) for the method, measured gains, and remaining
small-global-reduction regression.

For a reproducible noninteractive run:

```sh
cargo run --release -p fusor-storage-prototype -- --all --baseline --details
```

`--plan-only` performs allocation checks and Naga WGSL validation without a GPU.
`--dump-wgsl` prints the generated kernels. `--help` lists the remaining options.

## The proposed seam

```text
logical values + logical index mappings
                 │
       enumerate candidate regions
                 │
       prove workgroup ownership
                 │
       derive interference lifetimes
                 │
      color / pack virtual allocations ── budget exceeded ──> try another cut
                 │
       logical load/store accessor
                 │
        generic WGSL stage emission
```

`graph.rs` contains no physical strides, buffers, offsets, binding numbers, or
workgroup allocations. A view maps a result's logical index to an input's logical
index. Reshape composes indexing and emits no stage or allocation.

`plan.rs` selects a region and the number of workgroups that own it. Each group
owns a contiguous logical slice of each member. The planner checks that every
internal read stays inside the same group's slice. It then derives temporary
lifetimes and uses `storage.rs` to assign workgroup storage. That assignment feeds
back into region admission; it is not merely a cleanup after fusion.

Cheap single-consumer pointwise operations and reductions of at most eight
elements can forward scalar expressions through the same accessor. They need no
allocation or stage. Ownership checks follow these expressions to their actual
materialized dependencies. Shared producers and requested outputs remain stored.

The same allocator assigns a global arena to values escaping regions. Each
kernel has two storage bindings: inputs and the output/intermediate arena.
Pointwise and reduction emitters request logical loads/stores from `Access` in
`emit.rs`. Only that accessor knows whether a value lives in shared memory, the
global arena, or an external input buffer.

## Experiments

Keep the operation graph fixed and change storage policy:

```sh
cargo run --release -p fusor-storage-prototype -- --legacy --case reshape_reduce --shared-bytes 1024 --packing dedicated --details
cargo run --release -p fusor-storage-prototype -- --legacy --case reshape_reduce --shared-bytes 1024 --packing colored --details
cargo run --release -p fusor-storage-prototype -- --legacy --case softmax --shared-bytes 1024 --packing bestfit --details
```

Compare global memory reuse independently of fusion:

```sh
cargo run --release -p fusor-storage-prototype -- --legacy --case reuse --packing dedicated --details
cargo run --release -p fusor-storage-prototype -- --legacy --case reuse --packing bestfit --details
```

Exercise short reductions, tails, and different ownership partitions:

```sh
cargo run --release -p fusor-storage-prototype -- --all --rows 7 --cols 130
cargo run --release -p fusor-storage-prototype -- --all --no-fusion
```

The six cases cover a reshape between two reduction stages, explicit softmax,
fanout with two requested outputs, transpose across ownership partitions, a
global reduction, and reuse across five dispatches.

## What is and is not optimized

This enumerates **all contiguous fusion cuts in one topological order**, up to
12 compute stages. For each region it considers equal contiguous ownership
partitions whose group count divides every member's element count. The winning
plan minimizes an explicitly uncalibrated launch/traffic/work estimate under the
configured shared and global arena limits.

Both coloring and variable-size packing are deterministic heuristics. Reported
peak-live bytes are a lower bound for the chosen lifetimes. When allocated bytes
equal that bound, packing is optimal for those lifetimes. This does not prove
globally optimal fusion, scheduling, tiling, or runtime.

Constraints deliberately left visible:

- Static, nonempty shapes and f32 only; 64 lanes per workgroup.
- Ownership is proved by enumerating indices. A production compiler would use
  symbolic dependence analysis rather than paying work proportional to tensors.
  Finite access analysis is cached across candidate regions and checks source
  bounds. Its temporary host storage is also proportional to enumerated reads.
- Materialized intermediates use workgroup storage. Scalar forwarding is
  implemented; register tiling, general rematerialization, and schedule
  reordering are not implemented.
- A transpose cut is a limit of the contiguous ownership choices in this
  prototype. Different ownership, scalar forwarding, or staging can sometimes
  remove that cut. Allocation reuse by itself does not establish legality.
- Reductions up to 32 elements run serially per output lane; wider reductions
  use a shared-memory tree. Legacy mode uses a threshold of eight. There is no
  split reduction or tuned kernel family.
- Stage barriers are conservative and uniform. Storage live ranges include both
  a stage's reads and writes, and reduction-loop reuse includes a closing barrier.
- Native optimized compilation omits injected loop guards after excluding u32
  counter overflow for the generated counted loops. Buffer bounds checks remain
  enabled. Legacy mode retains the original fully checked compilation path.
- Inputs are externally supplied and excluded from the global arena budget.
  Output lifetimes extend through readback. This runner submits plans serially.

The existing compiler comparison uses the copied working-tree baseline and
reports dispatch count, first-resolve time, and reference correctness. Errors in
that optional comparison are printed explicitly and do not stop later prototype
cases. Prototype correctness failures stop the run. The prototype's batched
host time and Fusor's cold resolve time measure different things; they do not
establish a speedup.

The separate `--bench` path captures existing compiler command records through
the opt-in `fusor-gpu/prototype-capture` feature. This instrumentation is absent
when that feature is disabled, and it does not change production lowering. The
benchmark holds the graph alive and performs no further pool allocations until
replay completes; the capture API is not a general-purpose executable cache.

See [NOTES.md](NOTES.md) for the observed result and what to retain or delete
after this design experiment.
