# Fusor kernel visualizer

A deterministic virtual GPU for examining **where each lane accesses memory**, one
action at a time. Run it as a TUI, plain text, or newline-delimited JSON. It needs
no GPU or driver. The implementation is in the `fusor-visualize` workspace crate.

The repository includes selected measurement evidence. Full local experiment
archives are excluded; see [report contents](../../reports/PR-CONTENTS.txt).

For real Llama hotspot attribution, quantized-matrix experiments and full-logit
A/B validation, see the steady decode investigation (local experiment archive),
the prefill/latency investigation (local experiment archive)
and the original 2x investigation (local experiment archive).
Its `profile_kernel_share` example joins actual dispatches to semantic shapes,
layouts and GPU timestamps, including operations the virtual interpreter cannot
execute. The report includes reproduction commands and distinguishes prompt GPU
speedup from fresh-session latency and steady decode.
The decode report includes explicit GPU and wall speedup assertions. An earlier
normal WebGPU build passed **3.19× and 3.09×** at 4008-token context in ABBA
order: baseline medians 71.42/71.49 ms per token, candidate 22.39/23.12 ms,
60 steady samples per arm. The same executable runs both configurations;
timestamps, metadata and logit writes are disabled for these timings. This
build separately passes all 2,308,608 long-context logits (maximum absolute
difference 0.000471) and the cache-capacity transition at short context.
See `webgpu-rms-paired-wall-4008/result.json` (local experiment archive).
Reproduction uses `webgpu-rms-4k-config.json` and `webgpu-rms-profile` under
`reports/llama-decode-2x/`.

The original **208-token target now passes 2× in both orders**: **2.037× / 2.004×**
(32.632/32.042 → 16.020/15.990 ms/token), or **2.020×** using the median of
run medians. These are 62 steady samples per arm on the M2 Max, using the same
executable and resource policy. All 2,308,608 original-reference logits pass;
maximum absolute error is 0.0000611, with matching argmaxes and text. This is
the October 12 opt-in checkpoint, with a narrow margin in the reverse order.
Those runs retained at least 8.645 GiB estimated reclaimable headroom and
produced no additional swapouts. Quantized weights and F32 arithmetic remain.

The ordinary Llama builder now selects this inference policy on supported
Apple GPUs, without a profiling feature or environment preset. See
[normal chat usage](../../models/kalosm-llama/README.md#fast-chat-defaults) and
[default-path validation status](../../reports/llama-decode-2x/default-inference.txt).
The fresh full-model rerun was stopped by the memory guard during reference
prefill; the timings above are historical, not a new default-path measurement.

Use the verified configuration (local experiment archive)
and reproduction report (local experiment archive).
Exact timing evidence (local experiment archive)
and full-logit checks (local experiment archive)
are separate. The configuration includes a 32 ms active WebGPU polling window,
which trades CPU activity for lower completion latency. See
the investigation history (local experiment archive).
Historical native Metal and indirect-command results are excluded from accepted
speedup claims; those experiments are archived and are not build inputs.
The cache ownership and residual-normalization experiments are
recorded in `reports/llama-decode-2x/continuation-findings.txt`; a faster GPU span
alone does not establish a faster token. Cache donation has passed all model
logits and shared-buffer snapshot tests; combined with residual normalization
it modestly improves on the earlier 1.60× prompt-sized cache checkpoint.
`reports/llama-decode-2x/wall_ab.py` excludes timestamps, metadata, and logit
writes and checks equal prompts, phases, executable identity, and generated text.
The earlier `packed-candidate.json` and `retained-*` reports are historical.

```sh
cd fusor
cargo run -p fusor-visualize -- --demo strided --subgroup-size 32 --format tui
cargo run -p fusor-visualize -- --demo coalesced --subgroup-size 32 --format text
cargo run -p fusor-visualize -- --demo reuse --subgroup-size 16 --cache-bytes 128
cargo run -p fusor-visualize -- --demo bank-conflict --format text
cargo run -p fusor-visualize -- --demo padded --format text
```

TUI controls: **n / right / space** steps forward; **p / left** steps backward;
**r / home** resets to a cold cache; **e / end** finishes; **j/k / up/down** scrolls;
**PageUp/PageDown** pages; **q / escape / Ctrl-C** exits. The terminal is restored
on normal exit and error. Step back deterministically replays from the beginning,
so it has linear cost in the cursor position. Resize is supported. Long lines
are clipped horizontally; use text/JSON for the full source path or diagnostic.

The current action shows active lanes, byte addresses, sector/bank mappings,
transaction counts, line hits/misses, evictions, and opportunities to investigate.
Cumulative statistics remain visible at the bottom of the TUI.

## Compare layouts

With the default 32-lane subgroup and four-byte elements, these examples produce:

| Demo | One subgroup's memory pattern | Result |
| --- | --- | --- |
| `coalesced` | `address = lane * 4` | 4 sectors; packed lower bound 4 |
| `strided` | `address = lane * 128` | 32 sectors; packed lower bound 4 |
| `broadcast` | all lanes read address 0 | 1 sector; 4 unique bytes, 124 duplicate read bytes |
| `reuse` | every subgroup reads the same 32 elements twice | reuse across subgroups and cache hits |
| `bank-conflict` | shared index `lane * 32` | 32 bank rounds; lower bound 1 |
| `padded` | shared index `lane * 33` | 1 bank round; lower bound 1 |

These are small actual `KernelIr` programs. Change `--subgroup-size` to 16, 32,
64, or another width to regenerate their addresses and subgroup assignment.
`--workgroup-size` defaults to 64 and allows partial subgroups. `--workgroups 2`
captures both demo workgroups to expose reuse between them.

```sh
# Scriptable, deterministic output, including configuration and final totals.
cargo run -p fusor-visualize -- --demo reuse --format json > reuse.jsonl

# Export is create-new: choose another filename if it already exists.
cargo run -p fusor-visualize -- --demo reuse --export-trace reuse.trace.json
cargo run -p fusor-visualize -- --trace reuse.trace.json --format tui

# A small portable trace with no subgroup-dependent address generation.
cargo run -p fusor-visualize -- \
  --trace fusor-visualize/examples/trace.json --subgroup-size 2
```

`--max-steps N` bounds text/JSON playback and labels an incomplete result as a
prefix. `--max-actions N` bounds IR capture, including empty loop iterations and
expression evaluation. Exceeding that budget returns an error without a partial
trace. JSON input is limited to 32 MiB by the CLI. Library callers own their input
and memory budgets. Run `--help` for all options.

## Capture a Fusor kernel

The integration seam is **after GPU lowering produces `KernelIr`**. The recorder
is independent of the GPU backend and planner. Call it from a compiler debugging
tool or test that already has the lowered kernel; no production execution hooks
or environment variables are installed by this crate.

```rust,ignore
use fusor_visualize::{kernel::{capture, CaptureOptions}, GpuConfig, Simulator};

// `ir: fusor_ir::ir::kernel::KernelIr` from the lowering under investigation.
let trace = capture(&ir, CaptureOptions {
    subgroup_size: 32,
    workgroups: vec![[0, 0, 0], [1, 0, 0]], // must be inside ir.grid
    max_actions: 100_000,
    ..CaptureOptions::default()
})?;
// Save this JSON for the CLI, or drive the simulator in the current process.
serde_json::to_writer_pretty(std::fs::File::create("kernel.trace.json")?, &trace)?;
let mut gpu = Simulator::new(trace, GpuConfig::default())?;
while let Some(step) = gpu.step() {
    println!("{}", fusor_visualize::render::step(&step, gpu.trace(), gpu.config()));
}
```

The current adapter supports scalar storage/workgroup accesses, masked
loads/stores, atomic-add address traffic, lane/grid/subgroup builtins, local
variables, integer address arithmetic, comparisons, select/cast, bounded loops
with simultaneous accumulator updates, branches, return, barriers, and subgroup
reduction input accesses, quantized loads through the compiler's exact block
decoder, and reads inside vector arithmetic (whose computed values remain
unknown). Rank-2 addresses use Fusor's `MultiFlattenMap`, including
divmod chains, broadcast strides and view offsets. Sources name the exact IR path
(for example `body[2].loop[0].value.left`).

This is **address analysis**, not a numerical kernel interpreter. Loads produce
unknown values unless their binding bytes are supplied in
`CaptureOptions::known_bindings`. This resolves runtime uniform dimensions and
concrete input indices. Writes conservatively invalidate the whole binding's
known contents. Unknown values can flow through arithmetic and stores. When such a value
controls a branch, loop count or address, capture fails with a diagnostic. Unary
math, rounding, half literals, and subgroup reduction results also remain unknown.

Collective fills, workgroup reductions, cooperative matrices, vector memory
loads/stores, unbounded loops/break, and byte-arena aliasing are rejected
before capture, including in unreachable branches. Their concrete memory
instructions depend on backend lowering. A post-emission lane-address trace can
represent their actual accesses without adding guesses to this adapter. There is
no WGSL parser or post-emission instrumentation included in this first version.

The quantized recorder expands the same block program used by GPU emission
and traces its scalar word loads in the selected Native or F32Scales layout.
The matrix lab reports `LAB_SELECTED_QUANT`; requesting Native input does not
guarantee Native execution because the compiler can select a lossless scale
repack. `LAB_TRACE_DIR` exports actual matrix-kernel traces and virtual-cache
summaries.

IR recording evaluates **each explicit expression use, before backend CSE and
vectorization**. Repeated logical loads may be eliminated by emission. Treat IR
counts as opportunities to inspect, not as measured machine instructions. An IR
trace records its subgroup size and rejects replay at a different size: regenerate
it from the kernel so subgroup-dependent addresses stay correct.

Load masks describe **logical accesses**: masked-out lanes emit no trace load.
The current GPU storage emitter can instead load a clamped address on every lane
and then select the fill value. Consequently this adapter can also undercount
emitted reads. `Select` expressions themselves evaluate both arms in the recorder.
Inspect generated WGSL before concluding that a logical inactive lane performs no
memory instruction; final machine-code optimization can change the traffic again.

## Design and model assumptions

```text
KernelIr + capture options       external lane-address trace
          |                                 |
    scalar address recorder                 |
          +------------ Trace v1 -----------+
                          |
                 Simulator::step / seek
                 configuration + cold state
                          |
                 Step + metrics + hints
                     /           \
                text / JSON       TUI
```

The simulator owns address grouping, reuse attribution, cache state, bank
analysis and cumulative totals behind `new`, `step`, `seek`, and `reset`. Renderers
only present its output. This gives both frontends identical behavior and keeps
the tests at the same interface used by callers.

One action is a memory instruction issued by one subgroup, or a workgroup control
action/barrier. A trace memory event contains all participating workgroup lanes;
the simulator splits it into contiguous subgroups in ascending order. Empty masks
issue nothing. Inactive lanes do not access memory. Workgroups run in capture
order on one virtual SM. There is no latency, occupancy, concurrent scheduling,
L2/DRAM, register-pressure or instruction-throughput model.

| Setting | Default | Meaning |
| --- | --- | --- |
| Subgroup size | 32 | consecutive local invocation indices per subgroup |
| Transaction bytes | 32 | aligned storage sector size |
| Cache line bytes | 128 | read allocation/probe granularity |
| Cache bytes | 16384 | one shared fully associative LRU cache; 0 bypasses it |
| Banks | 32 | independent shared-memory banks |
| Bank width bytes | 4 | bytes in a bank word |

All resources have independent, aligned byte-address origins. Aliased storage
views must use the same resource ID. Workgroup resources are private to each
workgroup. IR storage views sharing a binding share an ID; tiles get separate IDs.

**Coalescing:** count the union of aligned sectors touched by all active byte
ranges of one subgroup action. Accesses crossing sector/line boundaries count
both. The packed lower bound is `ceil(unique bytes / sector bytes)`; it is an
optimistic packing comparison, not a promise that a legal layout achieves it.
Sector utilization is `unique bytes / (sectors * sector bytes)`. Duplicate bytes
are not counted as extra utilization. Sectors are address transactions, **not
DRAM traffic**; cache hits do not change the coalescing count.

**Caching:** one probe per distinct line per subgroup action, in ascending line
order. Read misses allocate a full line and evict the least recently used line.
Stores/atomics use a simplified write-through/no-allocate policy and invalidate
touched lines; they do not add read hits/misses or cache-fill bytes. Atomics count
address traffic, not read-modify-write traffic or serialization. Cache bypass
reports misses but no fills or evictions. Cache capacity and policy describe this
virtual GPU, not a specific vendor's hardware.

**Reuse:** unique read bytes are attributed to the most recent reader, distinguishing
same-subgroup, other-subgroup/same-workgroup, and other-workgroup reuse. Writes
invalidate the touched bytes' history. Duplicate bytes within one instruction
are reported separately. Reuse history is independent of residency: a reusable
value can miss the cache after eviction. Subgroups do not have fictitious private
caches. Suggested staging/broadcasts must repay their own instructions and barriers.

**Shared memory:** bank is `(byte_address / bank_width) % bank_count`. A step's
bank rounds are the maximum number of distinct words mapped to any bank; read
broadcasts to one word collapse. The lower bound is
`ceil(total distinct words / bank_count)`. Excess rounds flag imbalance, so 64
lanes evenly spread over 32 banks are correctly shown as two necessary rounds.
This aggregate model does not reproduce hardware instruction splitting. Overlapping
non-atomic writes get a possible-race diagnostic, not a broadcast optimization.

**Synchronization:** workgroup barriers advance the shared-memory visibility
epoch; storage barriers do not. A shared read from another subgroup's write in
the current epoch gets a synchronization hint. Reads with no prior explicit write
also get a hint; implicit zero initialization is not traced. Divergent workgroup
barriers in captured IR are errors. These checks are local diagnostics, not a race
detector or proof of memory-model correctness. Same-subgroup hazards, write/write
races across separate actions, and inter-workgroup races are not exhaustively checked.

The coalescing, bank mapping and read-broadcast concepts are based on the
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations).
All configurable values, cache policy, scheduling and subgroup assignment here
are explicit modeling choices, not claims that every GPU uses that organization.

## Portable trace format

`examples/trace.json` is a complete minimal example. `Trace` and its nested types
are serde serializable; unknown fields are rejected. IDs and addresses are unsigned,
addresses are in **bytes**, and each memory event belongs to one resource and one
workgroup. An access is `read`, `write`, or `atomic`; lane accesses specify `lane`,
`address`, and `bytes`. Resource sizes validate every access before replay.
`recorded_subgroup_size: null` allows regrouping a generic address trace; use a
number when the producer's addresses depend on that subgroup width. Do not use
null to regroup subgroup-dependent instructions.

Future backend integration should produce this same trace after concrete memory
instructions are known, carrying the originating IR path. That would support
collective/quantized kernels and post-CSE traffic without changing either frontend
or the memory model. Data-dependent control requires concrete execution inputs
or instrumentation rather than an inferred access pattern.

## Sweep a local Llama model

The opt-in `kernel-visualize` feature in `kalosm-llama` provides an inference
example that observes every distinct uncached GPU lowering, including compiler
candidates, and passes its packed runtime uniforms to the recorder:

```sh
# From the repository root; use a new output directory for each run.
mkdir -p reports
cargo run -p kalosm-llama --example visualize_kernels \
  --features kernel-visualize -- /path/to/model.gguf reports/llama-kernels 4
```

It exercises short and longer prefill plus decoding through the actual model,
deduplicates by IR and uniform values, and analyzes the first and last workgroups
of every observed variant. `report.json` contains the inference run status,
complete observed inventory, per-kernel sampled totals, hotspot source paths,
and unsupported reasons. `submitted_dispatches` separately counts successful
nonempty `GpuTarget::resolve` submissions of an exact body/grid/uniform variant.
`submitted_variants` also includes warm variants that were never relowered.
These counts include tuning runs and do not establish the final winning plan.
`kernels/` contains bounded IR dumps and replayable
traces for supported variants. An inference failure retains a partial report
and exits unsuccessfully. Traces may contain supplied input/shape information;
this example supplies only uniform words, not model weights or token buffers.

Trace bounds failures are reported separately: logical IR addresses can differ
from emitted accesses after backend clamping/robustness, so these failures do not
by themselves establish an out-of-bounds GPU access. To generate a human-readable
summary with every variant and ranked example sources:

```sh
python3 fusor/fusor-visualize/examples/summarize_llama.py reports/llama-kernels/report.json
```

This is coverage of the specified model, prompts and decode steps, not every
possible context length or quantization format. Observation counts are compiler
events, not dispatch counts. Cached artifact hits are not relowered. Metrics are
for sampled workgroups, not scaled estimates of full-grid traffic. GPU inference
is needed to discover these kernels; replay remains entirely CPU-based.

The underlying `fusor-gpu/kernel-observer` feature exposes
`GpuTarget::set_kernel_observer` and `set_dispatch_observer`. It adds no observation work to normal builds,
does not depend on the visualizer, and invokes callbacks outside its lock on
compiler workers. Callbacks should only collect snapshots; the example performs
analysis after inference.

For focused inspection, optional positional arguments select comma-separated
body-hash prefixes and 1–32 adjacent workgroups at each end of the grid:

```sh
FUSOR_WGSL_DUMP=reports/llama-focused/wgsl \
cargo run -p kalosm-llama --example visualize_kernels \
  --features kernel-visualize -- /path/to/model.gguf reports/llama-focused \
  4 481141fe3bada9eda021280fd023a22c 32
```

The two sampled windows omit intervening workgroups. Cache results depend on
that order; they are not a hardware scheduling prediction. The chunk count is
emitted text chunks, which need not be one token each.

## Validation

```sh
cargo test -p fusor-visualize --offline
cargo clippy -p fusor-visualize --all-targets --offline -- -D warnings
cargo fmt -p fusor-visualize -- --check
```

The regression cases exercise numerical transaction counts, misalignment,
cross-line accesses, broadcast, cache LRU order and bypass, resource identity,
write invalidation, reuse scopes, partial subgroups, bank padding, shared-memory
isolation, barriers, deterministic rewind, trace validation, and the IR recorder's
masking, loop state, control flow and failure behavior.


### Quantized weight analysis

`tools/quant_entropy.py MODEL.gguf --output entropy.json` samples native Q4_K
and Q6_K coefficients with NumPy. It reports a zero-order histogram coding
scenario per tensor, retaining original scale metadata. This is not a codec,
a bound on conditional compression, or evidence of achievable GPU bandwidth.


### WebGPU weight-layout probe

`fusor-gpu/examples/q4_layout_lab.rs` compares original Q4_K words with a lossless
interleaving across output columns. It varies lanes per column, rotates weight
buffers, changes fractional F32 inputs, alternates arm order, checks every output
against CPU dequantization on the first input, compares all outputs between arms,
and checks NaN tail canaries. Shader runtime checks stay enabled. Reported times
are isolated kernel spans, not model decode speedups.

```sh
cd fusor
cargo build -p fusor-gpu --example q4_layout_lab --offline --config profile.dev.opt-level=3
# Tail/indexing check, then the Llama gate/up shape.
target/debug/examples/q4_layout_lab 257 1024 2 8
LAB_LAYOUT_UNROLL=1 LAB_LAYOUT_STATIC=1 target/debug/examples/q4_layout_lab 28672 4096 8 8
```

`LAB_LAYOUT_LANES=1,2,4,8,16,32` selects lane counts. `LAB_LAYOUT_UNROLL` emits
fixed word iterations; `LAB_LAYOUT_STATIC` gives storage arrays exact extents.
`LAB_LAYOUT_GROUPED` shares scale arithmetic across words; `LAB_LAYOUT_MASKED`
uses positioned nibbles; `LAB_LAYOUT_PAIRED` shares inputs across two columns.
`LAB_LAYOUT_BLOCKS=1` preserves the production mapping of four K blocks and two
columns per subgroup while transposing words within each four-block packet.
With that mode, `LAB_LAYOUT_INPUT=uniform` or `shared` instead compares two
ways to read the same activations and leaves the weight layout unchanged.
`LAB_LAYOUT_SUBGROUPS=2` or `8` selects 64 or 256 threads. The 512-thread probe
failed its output check on this adapter and is not an accepted configuration.
Each paired timing is retained in stderr as `Q4_LAYOUT_SAMPLE`.

`LAB_LAYOUT_SHADER_BASELINE` and `LAB_LAYOUT_SHADER_CANDIDATE` can supply
captured WGSL for a block/input-mode comparison. They must match the requested
shape, three bindings (weights, input, output), and dispatch geometry. The same
CPU checks and canaries apply; shader compilation still uses ordinary WebGPU
runtime checks. This exposed a 32% checked-shader gain from vector activation
loads, but the corresponding compiler experiment was flat with the runtime's
existing compilation policy. That diagnostic gain is **not** a production or
model speedup. Evidence is in
`webgpu-q4-production-uniform` (local experiment archive)
and `webgpu-vector-input` (local experiment archive).

`LAB_LAYOUT_CORRUPT_COLUMN=129` intentionally corrupts a scale for a negative
control; the check must fail. The initial looped variants are slower than the
production packed kernel, so interleaving alone is not an accepted optimization.

`fusor-gpu/examples/q6_layout_lab.rs` compares the current 212-byte Q6 blocks
with separated bit/scale planes or 256-byte aligned blocks. Both representations
preserve the quantized values and F32 arithmetic. It shares the CPU reference,
changing-input checks and tail canaries with the Q4 tool; all GPU outputs must
match between arms bit for bit. Each timing is recorded as `Q6_LAYOUT_SAMPLE`.

```sh
cargo build -p fusor-gpu --example q6_layout_lab --offline --config profile.dev.opt-level=3
LAB_Q6_LAYOUT=planes target/debug/examples/q6_layout_lab 257 1024 2 8
LAB_Q6_LAYOUT=aligned LAB_Q6_SPLITS=2 target/debug/examples/q6_layout_lab 4096 14336 16 16
```

`LAB_Q6_SPLITS=1` or `2` selects direct or split reduction;
`LAB_Q6_CORRUPT_COLUMN=129` is the failing negative control. The checked-shader
layout gains were small. The actual compiler trial of aligned blocks regressed
and was removed. See `webgpu-q6-layout` (local experiment archive)
and `webgpu-q6-alignment-format` (local experiment archive).

The two-resident-process controller `tools/pair_qmatmul.py` is disabled.
The related full-model `interleaved_wall.py` experiment exhausted memory on
the 32 GiB development machine, before producing a valid comparison. Serialized
GPU execution does not bound the memory occupied by two loaded models. Both
controllers are archived as source text in
`excluded-resource-experiments` (local experiment archive).
Existing microbenchmark results remain historical evidence. The replacement
`tools/sequential_qmatmul.py` runs A, B, B, A in four sequential processes,
through the same pressure guard as the model harness. It limits resident weight
copies to 512 MiB, caps the GPU pool at 3 GiB, records executable/configuration
provenance and verifies every value of three changing-input output vectors.
`--weight-file` uses native blocks extracted from the actual model.

To use actual model data, `tools/extract_quant_weights.py` copies named GGUF
matrices into a new raw block file and records their offsets and SHA-256.
Matrices with matching format and K can be concatenated along output rows:

```sh
python3 fusor-visualize/tools/extract_quant_weights.py model.gguf \
  blk.0.ffn_gate.weight blk.0.ffn_up.weight --output gate-up.bin
```

`qmatmul_lab` reads these bytes through `LAB_WEIGHT_FILE`; CPU checks
use the original native blocks before any GPU layout conversion.

`reports/llama-decode-2x/paired_wall.py` wraps the clean full-model harness in
ABBA order and requires the speed target in both orders. It verifies the same
executable, configurations, prompt, phases, and generated text across all four
sequential processes. Background GPU workloads can still affect timings.
Full-logit correctness is checked separately.

The full-model launchers `wall_ab.py`, `probe.py`, and `model_ab.py` now use
`tools/benchmark_guard.py`. The guard holds a shared, per-user lock until its
child has exited, refusing overlapping guarded runs even across worktrees.
Before spawning, it requires at least 12 GiB of estimated reclaimable memory
(or twice the model size plus 6 GiB, whichever is larger), 8 GiB of disk space,
and at most 2 GiB in the compressor. From process startup it polls every 0.5 s
and aborts if memory headroom falls below 6 GiB, disk space below 4 GiB,
compression grows by 512 MiB, or swapouts grow by 64 MiB. Monitor errors,
interruptions, and timeouts also abort; it terminates only its own process
group, reaps the child, records the failure, and never automatically retries.

These are conservative pressure safeguards, not an OS-enforced GPU allocation
quota. Direct binary launches and historical microbenchmark scripts are not
covered. Use the guarded full-model launchers for future model measurements;
do not run a build or another GPU experiment alongside them. Resource logs sit
beside each run's output. Monitoring adds host activity, so future comparisons
must use the same guard for both arms. The first guarded width-32 comparison
measured 31.853 versus 18.636 ms/token (1.709×), with no swap growth, but later
correctness runs still hit pressure limits. Mapped GPU uploads alone did not
remove the retained private host weight copies.

The current bounded configuration uses one compiler thread, 64-token prefill
chunks, a small cooperative tile, mapped initial uploads, and a 7 GiB GPU pool.
`PROFILE_MMAP_WEIGHTS=/absolute/path/model.gguf` retains read-only source views
after checking metadata and every byte against the loaded tensors. The file
must remain unchanged while the model is alive. Concatenated quantized matrices
retain host segments and materialize only one upload/repack at a time. The
208-token prompt, F32 activations and accumulation, and original quantized
weights are unchanged. Both timing arms must use the same resource settings.

The full two-session check in
`segmented-host/logits` (local experiment archive)
passes all 2,308,608 logits (maximum absolute difference 0.0000639), with at
least 10.69 GiB estimated reclaimable memory, no compression growth, and no
swap growth. `tools/check_model_logits.py` checks every output without GPU
timestamp instrumentation. This establishes correctness and bounded execution
for that historical run, not a new speed result. At that checkpoint the short-context
2× goal remained open. A corrected ABBA comparison with that
resource policy measured 32.314/32.217 versus 18.808/19.158 ms/token
(1.718×/1.682×). See `segmented-host/scoped-original-wall`.

The extra prefill limits now use
`FUSOR_PROFILE_LARGE_PLAN_MIN_DISPATCHES=4096`, independently of normal pass
chunking. The earlier threshold of 1024 also throttled the original
1270-dispatch decode baseline; its apparent 2.4–2.7× result is excluded.
The wall harness rejects comparisons whose decode phases cross the prefill
resource threshold. Use `baseline-scoped.json` and `retained-scoped.json`.

Kernel microbenchmarks must also match the model’s shader options, including
`FUSOR_PROFILE_SKIP_INITIALIZED_WORKGROUP_ZERO`. With matching options, the
previous Q6 float-unpack micro gain shrinks to 1.7–2.0%; actual model Q6 down
and vocabulary timings remain essentially unchanged. The retained configuration
excludes that experiment. Shared backing and segmented concatenations pass
focused CPU/WebGPU tests, and all 14 Python tool checks pass.

Read-only preflight and lightweight regression checks (neither loads a model):

```sh
python3 fusor/fusor-visualize/tools/benchmark_guard.py --model /path/to/model.gguf
python3 -m unittest discover -s fusor/fusor-visualize/tools -p test_benchmark_guard.py
```

## Bounded geometry comparisons

`tools/paired_geometry.py` compares two legal SGEMV schedules inside one actual
compiled WebGPU plan. It alternates ABBA/BAAB, verifies three changing full
output vectors, and limits resident weights to 512 MiB. `--dependent` inserts
an identical checked WGSL output-to-input dependency guard after each GEMV;
its timings include the guard and still do not represent model throughput.
A/A control is within 0.2%. Four Q4 columns per subgroup improves this microprobe
about 4.6%, but the separate model ABBA comparison is flat (0.997× / 1.008×).

`tools/paired_model_geometry.py` uses one loaded model and two small KV histories,
alternating the execution order of matching tokens. It never replays stateful
plans. The schedule override rejects changes to graph extraction, launch order,
buffers, symbols, and illegal schedules. Full-logit mode checks both histories
against the original reference. It passed 4,617,216 values, but a three-round
control exceeded the compressor-growth guard. The tool now allows only one
warmup and one measured round, at most 32 steps. That capped A/A control passes
(0.99993× / 1.00332×), with no compressor or swap growth. The earlier aborted
timing remains invalid. The harness also supports bounded polling and submission
policy comparisons without loading a second model.
All probes use `benchmark_guard.py`; a resource abort is not retried automatically.
