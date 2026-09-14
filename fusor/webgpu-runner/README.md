# Browser transformer

Open `#/train` in the release runner. The Architecture & training panel controls
blocks, residual width, attention heads, feed-forward width, context, batch size,
and the training token budget. Apply changes to start with fresh weights; pause
first to change a running model. Resume keeps its weights, optimizer and schedule.

The default is 3 blocks, width 96, 4 heads, feed-forward width 192, context 128,
and batch 16: 248,352 parameters and 2,048 character tokens per step. A requested
20,000,000-token run takes 9,766 complete steps (20,000,768 tokens). Warmup lasts
up to 40 steps, then the learning rate decays from 0.003 to 0.0003 across the run.
Optional early stopping uses held-out loss; it is off by default. Token counts
include repeated sampling of training windows, not unique corpus coverage.

The [TinyStories slice](assets/TinyStories.md) contains 7,999,444 character
tokens across 9,804 complete stories, about 25 times the previous corpus. A split
between stories near 90% reserves the tail for evaluation. The first visit downloads it
separately from an immutable snapshot; SHA-256-verified bytes are cached locally.
It is neither tracked in the current source tree nor embedded in Wasm. A cached
copy works without corpus network access; an initial download failure shows a
retry control. Native examples use curl and a filesystem cache (override the
system temporary directory with `FUSOR_CORPUS_CACHE`).

`ModelConfig` is shared by training, generation, parameter counts and attention
inspection. Validation runs before device creation: widths must divide evenly
among heads, dimensions must be in range, and the combination must fit a
conservative 256 MiB working-memory estimate. This admission limit does not
guarantee allocation on every adapter; actual GPU allocation/compilation errors
are reported by the page.

The Tiny preset retains the old architecture. Compiler benchmarks also use the
original 65-character vocabulary and corpus via `Corpus::benchmark().await`, so their
240,480-parameter workload, loss oracles and timing comparisons stay unchanged.
The browser Tiny preset uses the expanded corpus and its 74-character vocabulary.

## Validation

From `fusor`:

```sh
cargo test --offline --release -p fusor --example train_small -- --test-threads=1
```

The model checks compare all parameters and Adam moments with the reference
executor for the original shape, the new default, and an irregular custom shape.
They also check held-out evaluation, generation, embedding similarity and causal
attention maps.

With a release runner on port 8900, Playwright on `NODE_PATH`, and a WebGPU-capable
Chrome executable in `CHROME`, run from this directory:

```sh
node tests/configuration.cjs
node tests/corpus-cache.cjs
```

This drives the production UI: invalid configurations, a custom architecture,
budget rounding and stopping, generation, attention images, reset, pause/resume,
and a full default 20-million-token run. `FUSOR_URL` overrides the address;
`FUSOR_ARTIFACTS` saves results and desktop/mobile screenshots. Optional compiler
capability checks in `tests/training.cjs` require `--features training-checks`.
`FUSOR_PRODUCTION=1` asserts that these diagnostic exports are absent;
`FUSOR_QUICK=1` shortens the final UI smoke run to 256 steps.

Measured on an Apple M2 Max in Chrome 152, the default full UI run completed in
72.1 seconds including model compilation, periodic evaluation and sampling, plus
one pause/resume. Training averaged 5.6 ms per step, ending at 0.912 held-out loss
and 72% next-character accuracy. This is a different workload from the original
tiny compiler benchmark: twice the tokens per step, twice the context and a
larger vocabulary. These are single-device measurements, not portable targets.

When another `dx serve` watcher is active, set a separate `CARGO_TARGET_DIR` for
this server. Dioxus writes generated JavaScript and Wasm into the target tree;
concurrent release builds can otherwise serve mismatched files.
