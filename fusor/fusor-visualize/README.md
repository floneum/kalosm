# Fusor kernel visualizer

A deterministic virtual GPU for examining **where each lane accesses memory**, one
action at a time. Run it as a TUI, plain text, or newline-delimited JSON. It needs
no GPU or driver. The implementation is in the `fusor-visualize` workspace crate.

The repository includes selected measurement evidence. Full local experiment
archives are excluded; see [report contents](../../reports/PR-CONTENTS.txt).

The shared compiler recognizes normalization and streaming attention from ordinary
algebra, forwards views, and lowers contiguous copies through WebGPU. Quantized
load selection depends on the format, layout, subgroup width, and device limits.
No model-specific execution profile or adapter-name preset is required.
See [shared compiler validation](../../reports/shared-compiler.txt).

The `profile_kernel_share` model example joins actual dispatches to semantic
shapes, layouts, and GPU timestamps. Earlier model performance reports describe
historical experimental checkpoints; their speedup numbers do not establish
performance of the shared compiler rewrite.

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
# Tail/indexing check, then a larger aligned matrix.
target/debug/examples/q4_layout_lab 257 1024 2 8
LAB_LAYOUT_UNROLL=1 LAB_LAYOUT_STATIC=1 target/debug/examples/q4_layout_lab 2048 4096 8 8
```

`LAB_LAYOUT_LANES=1,2,4,8,16,32` selects lane counts. `LAB_LAYOUT_UNROLL` emits
fixed word iterations; `LAB_LAYOUT_STATIC` gives storage arrays exact extents.
`LAB_LAYOUT_GROUPED` shares scale arithmetic across words; `LAB_LAYOUT_MASKED`
uses positioned nibbles; `LAB_LAYOUT_PAIRED` shares inputs across two columns.
`LAB_LAYOUT_BLOCKS=1` preserves the production mapping of four K blocks and two
columns per subgroup while transposing words within each four-block packet.
With that mode, `LAB_LAYOUT_INPUT=uniform` or `shared` instead compares two
ways to read the same activations and leaves the weight layout unchanged.
`LAB_LAYOUT_SUBGROUPS=2` or `8` selects 64 or 256 threads. Every configuration
must fit the adapter limits and pass the output checks.
Each paired timing is retained in stderr as `Q4_LAYOUT_SAMPLE`.

`LAB_LAYOUT_SHADER_BASELINE` and `LAB_LAYOUT_SHADER_CANDIDATE` can supply
captured WGSL for a block/input-mode comparison. They must match the requested
shape, three bindings (weights, input, output), and dispatch geometry. The same
CPU checks and canaries apply; shader compilation still uses ordinary WebGPU
runtime checks. These comparisons measure the supplied shader and geometry;
validate any proposed compiler change separately on the ordinary tensor path.

`LAB_LAYOUT_CORRUPT_COLUMN=129` intentionally corrupts a scale for a negative
control; the check must fail. Output checks run before timings are accepted.

`fusor-gpu/examples/q6_layout_lab.rs` compares the current 212-byte Q6 blocks
with separated bit/scale planes or 256-byte aligned blocks. Both representations
preserve the quantized values and F32 arithmetic. It shares the CPU reference,
changing-input checks and tail canaries with the Q4 tool; all GPU outputs must
match between arms bit for bit. Each timing is recorded as `Q6_LAYOUT_SAMPLE`.

```sh
cargo build -p fusor-gpu --example q6_layout_lab --offline --config profile.dev.opt-level=3
LAB_Q6_LAYOUT=planes target/debug/examples/q6_layout_lab 257 1024 2 8
LAB_Q6_LAYOUT=aligned LAB_Q6_SPLITS=2 target/debug/examples/q6_layout_lab 2048 4096 16 16
```

`LAB_Q6_SPLITS=1` or `2` selects direct or split reduction;
`LAB_Q6_CORRUPT_COLUMN=129` is the failing negative control. The
layout probes do not change the compiler's selected representation.

`tools/sequential_qmatmul.py` runs A, B, B, A in four sequential processes,
through the shared pressure guard. It limits resident weight
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

`qmatmul_lab M_CSV N K FORMAT ITERATIONS` accepts every supported format
(`q4_0`, `q5_0`, `q8_0`, `q4k`, `q5k`, `q6k`). Set `LAB_QLAYOUT` to
`native` or `f32scales`; fixtures and reference checks use the format's block
metadata. K must be a multiple of that format's block element count.

`qmatmul_lab` reads these bytes through `LAB_WEIGHT_FILE`; CPU checks
use the original native blocks before any GPU layout conversion.

`tools/benchmark_guard.py` bounds experiment startup and monitors memory,
kernel memory pressure, swap, disk space, and timeout. Compression is recorded
but does not imply exhaustion on its own. The macOS pressure probe must report
normal; unavailable or unknown readings fail closed. A per-user lock prevents overlapping
guarded runs across worktrees. It terminates and reaps only its own process group
on failure, and never retries a resource abort automatically. Direct binary
launches are not guarded. Do not run a build or another GPU experiment alongside
a model measurement.

`tools/check_model_logits.py` can still compare saved output vectors offline.

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
These are kernel measurements, not full-model throughput claims. All probes use
`benchmark_guard.py`; a resource abort is not retried automatically.

`--binary` selects the fixture executable and `--dump-wgsl` captures its shaders.
To compare compiler transformations without separate-process timing drift, pass
`--baseline-wgsl` and `--candidate-wgsl` from captures of the same fixture and
schedule. Both modules compile through checked WebGPU with the live fixture's
explicit binding layout; identical buffers and ABBA/BAAB order are retained.
This mode deliberately retains runtime checks in both shaders. It is a kernel
diagnostic and must be followed by a default-path model measurement.

### Compare default decode latency

`profile_forward` observes actual sampled tokens through a dedicated tracing
target. It reports every token gap, p50/p95/worst latency, token IDs and generated
text. UTF-8 text buffering does not affect the token count. Sampling is seeded
(`KALOSM_PROFILE_LLAMA_SEED`, default 0); model,
prompt, sampler, warmup, and round settings are recorded in the log. Incomplete
samples fail instead of producing a successful throughput result.

Use the same example source, local model file, and settings in the base and
candidate checkouts. Set `KALOSM_PROFILE_LLAMA_REPEATS=3` (or more), run each
process separately under the resource guard, and capture stdout. Compare with:

```sh
python3 fusor/fusor-visualize/tools/compare_decode.py baseline.log candidate.log
```

The guarded runner records binary provenance, strips inherited tuning switches,
and measures one binary at a time with the ordinary device/compiler defaults:

```sh
python3 fusor/fusor-visualize/tools/profile_decode.py --binary /path/to/profile_forward \
  --model /path/to/model.gguf --output /path/to/new-measurement-directory
```

Use `--tokens` for a longer stall/growth investigation, keeping the same value
in every arm. It defaults to 64 tokens, eight warmup tokens, and three rounds.
The token sampler stops at exactly warmup plus measured tokens. The first
measured interval starts at the final warmup token, excluding prefill.
Add `--minimum-speedup 2` to `compare_decode.py` to make the throughput goal an
explicit failing check. A bracketed comparison must reach the target against
both baseline runs. Add `--minimum-tps 60` to require 60 actual tokens/sec.
Older chunk logs remain comparable to each other but cannot satisfy a token
throughput target. Build both revisions with the same example and token trace
event; otherwise the observer rejects the missing or incomplete sample.

The comparison excludes the first round from both logs, retains every slow
sample, and refuses mismatched settings, generated token IDs, text, warmup boundaries,
or incomplete logs. It reports elapsed-time, p95, and worst-gap ratios for
matched output. This is a wall-time comparison, not a GPU-kernel estimate.

To expose changing background load, run baseline, candidate, then baseline
again, serially under the resource guard:

```sh
python3 fusor/fusor-visualize/tools/compare_decode.py baseline-before.log candidate.log \
  --baseline-after baseline-after.log
```

All three runs must have matching settings and output. This reports baseline
drift and the smaller speedup against the two baselines for each metric. It
does not select the slower baseline or discard latency spikes.
