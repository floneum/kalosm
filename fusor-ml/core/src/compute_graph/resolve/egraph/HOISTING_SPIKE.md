# Recognition-hoisting measurement spike

`recognize.rs` keeps recognition outside equality saturation because ingesting
the un-preshrunk graph for every generated token was believed to be expensive,
and because a generator that re-derives a matmul needs a structural window
deeper than the two-step horizon plan sharing cuts at
(`structural_memo.rs`, `WINDOW_STUB_DEPTH`). Both costs were predictions. This
spike measures them.

The knobs are measurement-only and default off:

| variable | effect |
|---|---|
| `FUSOR_SPIKE_HOISTING` | emit the `hoisting_spike_ingest` / `hoisting_spike_windows` ledgers; changes no decision |
| `FUSOR_SPIKE_NO_RECOGNITION=<n>` | skip the pre-ingest recognition sweep for resolves with at most `n` execution nodes |
| `FUSOR_SPIKE_WINDOW_DEPTH=<d>` | widen the structural window horizon from the built-in stub depth |

## Method

One process, the real decode trace from the verification baseline:

```
FUSOR_TRACE_RESOLVE=1 FUSOR_TRACE_RESOLVE_HOST=1 FUSOR_SPIKE_HOISTING=1 [knob] \
  cargo run --release -p fusor --example transformer -- \
  --steps 2 --progress-every 0 --trace-resolve --trace-host
```

Two training steps, an eval pass, then 400 generated tokens. The 399 identical
steady-state decode resolves (`node_count=667` before recognition, 77 kernels
per token) are the population; every number below is the p50 over those 399
unless stated otherwise. Every configuration was run twice: all counts are
byte-identical run to run and the p50 timings agree within 1%. A run with
every spike knob off reproduces the golden `resolve_egg_plans` line and
decode dispatch histogram, and all four configurations generate byte-identical
text.

`FUSOR_SPIKE_NO_RECOGNITION` is scoped by graph size because the un-preshrunk
form of a *training* step does not run at all: with recognition disabled
everywhere, the first training resolve dies allocating 8.59 GB more (20.57 GB
already live under a 22.91 GB in-flight cap) — the contraction's `[.., N, K]`
product materialized instead of contracted. Budget `1000` keeps recognition
for the two training resolves (3498 and 3265 execution nodes) and drops it for
every eval, prefill and decode resolve (693, 663 and 667 nodes), all of which
run to completion un-preshrunk.

## (a) Ingest cost of the un-preshrunk decode graph

| per decode token | recognized | un-preshrunk | delta |
|---|---|---|---|
| execution nodes ingested | 387 | 667 | +72% |
| provenances | 466 | 746 | +60% |
| e-nodes / e-classes | 454 / 454 | 734 / 734 | +62% |
| interned payloads / specs | 47 / 47 | 72 / 72 | +53% |
| recognition sweep | 320 µs | 0 µs | −320 µs |
| ingest | 379 µs | 606 µs | +227 µs |
| window capture | 480 µs (311 windows, 1.55 µs each) | 911 µs (724 windows, 1.26 µs each) | +431 µs |
| `optimize` phase total | 1879 µs | 3019 µs | +1140 µs |
| resolver host total | 4144 µs | 5494 µs | +1350 µs |

Ingest itself is not the bill, and the never-measured cost turns out to be the
smaller half: the sweep costs 320 µs and saves 227 µs of ingest, so the
pre-shrink is a 93 µs *net win* on ingest alone. The rest lands in extraction
(1415 µs → 2765 µs): +431 µs capturing 724 windows instead of 311, and
+692 µs generating, looking up and costing candidates over a graph 60% larger.

The extraction *result* is much worse, which is the finding that matters:

| per decode token | recognized | un-preshrunk |
|---|---|---|
| dispatch categories | `flash_attention 6, matmul_f32 19, merged_matmul 6, nary_direct 33, row_program 13` | `merged_row 6, nary_direct 70, row_program 69` |
| kernels | 77 | 145 |
| extractor cost `work` | 9.86e9 | 4.90e10 |

No matmul and no attention kernel survives. Today's generators do not
re-derive a contraction from `Elementwise(Mul) + Reduce(Sum)`, so removing the
sweep does not move recognition into the e-graph — it deletes it, and the
generic row/nary lowering picks up the pieces at 5x the arithmetic work and
1.9x the dispatches. (It is still correct: the 400 generated tokens are
byte-identical to the recognized run, which is the `recognize.rs` "slower, but
correct" claim, measured.)

A recognizer-to-generator port therefore has to pay all of the above *and*
carry the matcher logic into the generator set; the un-preshrunk ingest is the
floor of its cost, not its cost.

## (b) Window-depth widening

Recognition on, only the structural horizon changes:

| stub depth | unique windows | intra-resolve hits | capture | capture per window | `optimize` phase |
|---|---|---|---|---|---|
| 2 (shipped) | 64 | 247 | 480 µs | 1.55 µs | 1879 µs |
| 4 | 81 | 230 | 823 µs | 2.65 µs | 2265 µs |
| 6 | 101 | 210 | 1316 µs | 4.23 µs | 2846 µs |

Windows captured stays 311 at every depth — widening does not plan more, it
plans the same windows less shareably: unique windows grow 64 → 81 → 101 and
per-window capture grows 1.7x / 2.7x, for +386 µs and +967 µs on the per-token
`optimize` budget (+21% / +51%) and +8% / +22% on the resolver's host total.

Plan-store miss rate is unaffected: after the first decode resolve of a
process warms the device store (`store_misses` 6 / 14 / 29 at depth 2 / 4 / 6),
every one of the remaining 398 resolves reports `store_misses=0` with
`store_hits` equal to its miss count (64 / 81 / 101). The store is
device-scoped and in memory, so a new process re-warms it on its first token;
across two process runs the ledgers are byte-identical.

Extraction output is unchanged at every depth: `dispatches=177
bytes=165875972 work=9863870292` and the same dispatch histogram. The depth-2
horizon is not costing fusion quality on this workload, so widening is pure
loss here — its only justification would be a generator that needs to see
further, which is exactly the recognizer-to-generator case.

## Go / no-go contract

A recognizer-to-generator campaign may proceed only if, measured on this
trace with these ledgers:

1. the un-preshrunk ingest growth plus the widened capture cost fits inside
   the decode `optimize_phases` budget the ported recognizers vacate — i.e.
   per-token `optimize` does not exceed the shipped 1879 µs p50 (recognition
   320 µs + extraction 1415 µs, 45% of a 4.1 ms host resolve), and
2. the second and later decode resolves of a process report
   `store_misses=0` — a port that makes windows layer-unique loses the
   device store and pays generation on every token, and
3. the extracted plan is unchanged: `dispatches`, `bytes`, `work` and the
   dispatch histogram must still match the goldens.

As measured, none of the three holds today. The minimum viable port — drop the
sweep and widen the horizon to 4 so a generator can see a contraction's
factors through their views — costs 3019 µs measured un-preshrunk plus the
386 µs depth-4 capture delta measured separately, roughly 3.4 ms against a
1879 µs budget, and its generators lose every matmul and every attention
kernel. Until a spike shows otherwise, pre-ingest destructive placement is the
accepted end state, and this document is the evidence that it is a measurement
and not a preference.
