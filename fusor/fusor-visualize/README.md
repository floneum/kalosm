# Fusor kernel visualizer

A deterministic virtual-GPU memory model and CLI for inspecting lane-level accesses
from Fusor kernels. The simulator replays a trace on the CPU; demo and serialized
trace playback do not require a GPU. An optional Llama example captures lowered
kernels during GPU inference and saves replayable traces.

## Quick start

From the repository root:

```sh
cargo run --manifest-path fusor/Cargo.toml -p fusor-visualize -- \
  --demo coalesced --subgroup-size 32 --format text
cargo run --manifest-path fusor/Cargo.toml -p fusor-visualize -- \
  --demo reuse --format json > reuse.jsonl
cargo run --manifest-path fusor/Cargo.toml -p fusor-visualize -- \
  --trace fusor/fusor-visualize/examples/trace.json --subgroup-size 2
```

Use `--format tui` for step-by-step terminal playback. The TUI supports stepping,
rewinding, resetting the cache, and scrolling; text and JSON formats are suitable
for scripts. Run with `--help` for the available demos and options.

## Capture and replay

The library boundary is a lowered `KernelIr`. Capture records logical memory
accesses for selected workgroups; the simulator and renderers then consume that
trace without a GPU backend. Callers with a lowered kernel can use
`kernel::capture` with `CaptureOptions`, serialize the resulting trace, or
replay it directly with `Simulator`.

The optional `kernel-visualize` feature on `kalosm-llama` provides a model
example. It observes kernel lowering during Llama inference, saves a bounded
inventory of traces and a JSON report, then uses the virtual model to analyze
sampled workgroups:

```sh
cargo run -p kalosm-llama --example visualize_kernels \
  --features kernel-visualize -- /path/to/model.gguf /path/to/new-output-dir 4
python3 fusor/fusor-visualize/examples/summarize_llama.py \
  /path/to/new-output-dir/report.json
```

This example requires a usable GPU and a local GGUF model for capture. Replay and
simulation are CPU-only. The trace and report describe the selected model,
prompts, and observed workgroups; they do not contain model weights or token
buffers. Sampling the first and last workgroups is bounded inspection, not a
full-grid traffic estimate. Emitted chunks are not guaranteed to correspond
one-to-one with tokens.

## Interpretation and limits

Capture evaluates explicit IR expression uses **before backend CSE and
vectorization**. Repeated logical loads can be eliminated during emission.
Masks describe logical accesses, while an emitter may load a clamped address and
select a fill value. Trace counts are therefore not machine instructions or
physical GPU traffic; they are opportunities to inspect the lowered IR.

The simulator is an explanatory model, not a numerical kernel interpreter or a
hardware timing model. Subgroup assignment, scheduling order, sectors, cache
policy, and shared-memory banks are explicit assumptions. Its transaction,
reuse, cache, and bank metrics are not measured latency, DRAM traffic, or a
prediction for a particular GPU.

Capture performs address analysis. Loads generally have unknown values unless
known binding bytes are supplied; unsupported or data-dependent control can
cause capture to return a diagnostic. The model does not reproduce backend
lowering for all collective, vector, or aliasing operations. A backend-emitted
trace would be needed to describe concrete post-lowering accesses for those
operations.

The independent Rust tests cover capture, trace validation, simulation, and CLI
output. They retain a regression ensuring reuse/staging advice appears even on
cache misses; only the “cache hits already help” qualifier depends on observed
hits.
