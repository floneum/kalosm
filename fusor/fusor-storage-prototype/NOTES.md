# Answer from the prototype

Date: 2026-09-13. GPU: Apple M2 Max. This is a throwaway design experiment.

**Historical first-prototype results below.** The optimized defaults added later
are measured in [PERFORMANCE.md](PERFORMANCE.md). Add `--legacy` to the commands
below to reproduce the original emission, forwarding, and scheduling policies.
The final implementation also caches finite logical access analysis; that change
preserves legacy kernel plans while reducing planning work.

**Yes: pointwise, reduction, and reshape can be expressed without physical
layouts in the operation IR or layout-specific operation emitters.** This
prototype executes them through a logical accessor after choosing workgroup
ownership and allocating storage. Coloring and variable-size packing can change
which regions fit, so storage assignment participates in fusion selection.

**Allocation does not replace execution planning.** A planner must still decide
which workgroup owns each logical element and prove that internal dependencies
can be synchronized. The prototype makes that proof explicit and independent of
the allocation offsets. Its transpose examples need cuts under the restricted
ownership choices; other schedules or scalar forwarding could remove some cuts.

## What changed architecturally

The current compiler already packs step-local global intermediates in
[`fusor-cost/src/plan.rs`](../fusor-cost/src/plan.rs) and workgroup allocations in
[`fusor-tile/src/arena.rs`](../fusor-tile/src/arena.rs). The proposed simplification
is not the existence of a packer. It is the interface and decision order:

1. `Graph` describes logical values, indexing views, and computations.
2. `compile(graph, config)` explores regions and proves ownership.
3. `allocate(slots, policy)` assigns storage from interference lifetimes. The
   region must fit its workgroup budget, and the complete plan must fit its
   global arena budget.
4. `Access::load` and `Access::store` translate logical access to shared, global,
   or input addresses. Operation emitters use this one interface.

Reshape has no emitting stage or allocated buffer. The same operation emitters
work for fused and unfused plans and all three storage policies. A result can be
both a shared intermediate and an externally requested output, exercised by the
fanout case.

This is a sibling executable, not a modification of production lowering. The
worktree's baseline commit `bd5ae04ed` snapshots the user's current Fusor working
changes. The prototype's separate changes add this crate and its workspace entry.

## Storage policy changes fusion

For 16 × 128 inputs and a **1 KiB workgroup-memory limit**, dispatch counts are:

| Case | Dedicated allocations | Greedy color slots | Variable-size packing |
| --- | ---: | ---: | ---: |
| Reshape between reductions | 2 | 1 | 1 |
| Explicit softmax | 2 | 2 | 1 |
| Fanout, two requested outputs | 2 | 2 | 2 |
| Transpose | 2 | 2 | 2 |
| Global reduction | 2 | 2 | 2 |
| Five-stage global reuse | 5 | 5 | 5 |

The fused reshape/reduction chain needs 768 B with coloring or packing. The
one-dispatch softmax needs 1028 B with fixed-size color slots and 1024 B with
variable-size packing. A four-byte placement difference crosses the configured
capacity limit and removes a dispatch. This deliberately tight budget exposes
the decision boundary; it is not a claim that the device has only 1 KiB available.

For the five-stage reuse case, dedicated global allocations consume **40 KiB**;
both reusable policies consume **16 KiB**, meeting the peak-live lower bound.
Dispatch count remains five. With a 16 KiB global cap, packing succeeds and
dedicated allocation reports that no plan fits. Global budgets exclude inputs.

Raw runs: [dedicated](results/dedicated.txt), [colored](results/colored.txt),
[packed](results/bestfit.txt), [one-region colored softmax](results/colored-softmax-detail.txt).

## Comparison with the current compiler

With the default 2 KiB shared budget and variable-size packing:

| Case | Prototype dispatches | Current Fusor dispatches |
| --- | ---: | ---: |
| Reshape between reductions | 1 | 1 |
| Explicit softmax | 1 | 1 |
| Fanout, two requested outputs | 1 | Error |
| Transpose | 2 | 2 |
| Global reduction | 2 | 2 |
| Five-stage global reuse | 5 | 5 |

The baseline fanout error is `plan: group member %28 lowers at 32 lanes, the group
at 256`. The prototype produces both requested outputs correctly. The other
baseline cases pass the same eager numerical reference for their comparison
input. This is one concrete mismatch in the copied current design; it does not
establish the coverage of a replacement compiler.

The default examples otherwise match existing dispatch counts. These observations
support a simpler operation/storage boundary, not a broad claim of faster code.
The printed prototype timing batches many steps and includes host submission and
waiting; the baseline first-resolve time includes compilation. They are not
comparable speedup measurements. The planner's score is also uncalibrated.

Raw run: [default with baseline](results/default.txt).

## Validation performed

- Built the release executable using the workspace's existing Rust/WGPU stack.
- Ran six cases across six configurations: default, three packing policies at
  1 KiB, odd/tail dimensions 7 × 130, and unfused dimensions 3 × 10. Each compiled
  plan checked three deterministic inputs against the separate eager reference:
  **108 GPU comparisons passed**.
- After removing an unused shared-memory declaration from zero-allocation
  shaders, ran the reuse case with a zero shared-memory budget and a 16 KiB
  global budget: **three more GPU comparisons passed, 111 total**.
- Each correctness execution overwrites inputs, poisons the global arena with
  NaNs, and dispatches the plan once before checking every requested output.
  Pipelines and allocations are reused across inputs. This catches stale global
  results that repeated warmup dispatches could otherwise conceal.
- Checked allocations for overlap between interfering lifetimes and validated
  emitted WGSL with Naga on every executed case.
- Exercised the terminal's case, packing, memory, fusion, allocation-detail,
  return, and quit controls. The GPU and baseline runners were exercised through
  the noninteractive commands.
- Verified that an insufficient global budget returns an explicit failure.
- Ran package-scoped formatting and `git diff --check`.

Additional raw runs: [tails](results/tails.txt), [unfused](results/unfused.txt),
[zero shared storage](results/zero-shared.txt).

Reproduce from `fusor/` with `cargo run --release -p fusor-storage-prototype --`
followed by one of:

```text
--all --baseline --iterations 100
--all --shared-bytes 1024 --packing dedicated --iterations 100
--all --shared-bytes 1024 --packing colored --iterations 100
--all --shared-bytes 1024 --packing bestfit --iterations 100
--all --rows 7 --cols 130 --iterations 100
--all --rows 3 --cols 10 --no-fusion --iterations 100
--case reuse --shared-bytes 0 --global-bytes 16384 --details --iterations 100
--case softmax --plan-only --packing colored --details
```

## What this does not settle

The exhaustive search covers contiguous cuts in one topological order and a
small family of equal contiguous ownership partitions. Packing is heuristic;
meeting the peak-live bound proves optimal storage only for those fixed
lifetimes. There is no claim of globally optimal fusion, tiling, or runtime.

The finite ownership proof scans tensor indices; default planning took roughly
1–21 ms on these tiny graphs. This is unsuitable for production scale. The
prototype supports static f32 shapes, conservative barriers, and one generic
reduction strategy. It does not model register pressure, vectorization, bank
conflicts, occupancy, rematerialization, mixed types, split reductions, or
noncontiguous external buffers. Its workgroup-memory materialization of every
internal value can lose badly to scalar forwarding.

## What to retain after review

Retain the separation between logical indexing, ownership/scheduling, and storage
assignment, plus a single logical accessor for operation emission. Feed packed
resource requirements into candidate-region selection. Preserve explicit proofs
for barriers and ownership; buffer reuse cannot supply them.

A production follow-up should first apply that interface to one real
pointwise/view/reduction path and compare equivalent warm executions. Replace
finite enumeration with symbolic dependence checks and reuse existing arena
machinery where possible. Keep tuned reduction schedules behind the same logical
access interface. Once the design is accepted or rejected, absorb the useful
interface and delete this executable, terminal shell, and illustrative cost
model rather than shipping a second compiler.
