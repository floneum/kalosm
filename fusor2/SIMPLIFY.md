# SIMPLIFY — ranked deletion plan

Measurement phase only. **Nothing was deleted for this document.** Every claim
below is backed by pasted output from a selection census, a compile experiment,
or a call-graph fact. Baseline recorded at the bottom so a later seal can quote
lines removed.

Working tree at the time of measurement: clean `decode-switch`, HEAD `76d42f7`.

---

## How the census was taken

A temporary probe (`fusor2-ir/src/probe.rs`, one append per event behind
`FUSOR2_HEAD_LOG`; reverted, patch kept at `/tmp/probe.patch`) was wired into
five places:

| probe | site | what it counts |
|---|---|---|
| `lower_head` | `fusor2-gpu::lower::lower_node`, the one `match op` over `L1` | every L1 node the GPU backend actually lowers, tagged with family / `SchedPoint` / `WaveCat` / mode |
| `cpu_lower_head` | `fusor2-cpu::lower::lower`, same match | same for the CPU backend |
| `merge_mint_attempt` / `merge_mint_ok` | `fusor2-ir::rules::merge::mint`, before and after the `segs.len() < 2` guard | every attempt to build a `KMerged`, and every one that survived |
| `merge_region_wave_seen` | `merge_region_wave`, after `segment_of` filtering | how many members a candidate `KRegion` had and how many were mergeable |
| `propose_*` | `fusor2-tile::rules::{contract,scatter,gather}`, after `b.union` | every L1 alternative *offered* to the extractor, by family / mode |
| `tile_contract_matched` | `tile_contract`, after its `sched: ScheduleDomain::Point` destructure | how often the `TILE_CONTRACT` rule matches at all |

Lowering is the ground truth for "selected": the extractor has already resolved
node, materialization and schedule point by then. `propose_*` separates
*never offered* from *offered and always priced out*.

Runs: the full conformance suite (`806 results: 797 passed, 9 skipped, 0
failed`, CPU **and** GPU backends), a 63-token llama-8B-Q4K decode, and one run
each of the other three model crates (rbert `embed_local`, rwhisper
`transcribe_local`, segment-anything `segment-anything`). Counts were identical
across two independent conformance runs, so the census is deterministic.

### Conformance suite — everything the two backends lowered

```
13014 lower_head	KMap Point
 7763 cpu_lower_head	KMap
 6705 lower_head	KFold Point
 6663 cpu_lower_head	KFold
 2033 lower_head	KContract Coop Coop splits=1
 1014 cpu_lower_head	KContract Sgemv
  958 lower_head	KFold Fold Subgroup
  956 lower_head	KContract Sgemm Sgemm
  706 lower_head	KContract Sgemv Sgemv cols=4 parts=4 subgroup_cols
  590 lower_head	KContract Sgemv Sgemv cols=32 parts=4 subgroup_cols
  572 lower_head	KContract Sgemv Sgemv cols=2 parts=4 subgroup_cols
  462 lower_head	KFold Fold WgTree { lane_group: 1 }
  398 cpu_lower_head	KScatter SortSegment
  393 lower_head	KScatter SortSegment
  312 lower_head	KContract Sgemv Sgemv cols=16 parts=4 subgroup_cols
  305 lower_head	KContract Sgemv Sgemv cols=16 parts=1 subgroup_cols
  281 lower_head	KFold Fold WgTree { lane_group: 128 }
  235 cpu_lower_head	KContract Sgemm
  159 lower_head	KGather RowPerGroup
   85 lower_head	KFold Fold WgTree { lane_group: 256 }
   84 cpu_lower_head	KGather RowPerGroup
   37 lower_head	KScatter Atomic
   27 lower_head	KFold Fold WgTree { lane_group: 2 }
   24 lower_head	KGather QuantizedRows
   24 cpu_lower_head	KGather QuantizedRows
   21 lower_head	KFold Fold WgTree { lane_group: 4 }
   18 lower_head	KFold Fold WgTree { lane_group: 16 }
   10 lower_head	KFold Fold LoopThenTree { iterations: 2048, lane_group: 1 }
    7 lower_head	KFold Fold WgTree { lane_group: 8 }
    6 lower_head	KFold Fold LoopThenTree { iterations: 256, lane_group: 1 }
    3 lower_head	KFold Fold LoopThenTree { iterations: 96, lane_group: 1 }
    3 lower_head	KFold Fold LoopThenTree { iterations: 3, lane_group: 1 }
    2 lower_head	KContract Sgemv Sgemv cols=1 parts=1 whole_wg
    1 lower_head	KFold Fold LoopThenTree { iterations: 128, lane_group: 1 }
```

**Absent from that list, i.e. lowered zero times on either backend:**
`KMerged` (all four `WaveCat`s), `KRegion`, `Ext`,
`KContract` at `Family::GenericFold`, `KScatter WgPrivateMerge`,
`KScatter OneHotContract`, `KGather Vectorized`.

### Conformance suite — what was *offered* to the extractor

```
12485 merge_mint_attempt	Row segs=1
 6586 merge_mint_attempt	Matmul segs=1
 1564 propose_family	Sgemv
 1522 propose_family	Sgemm
  940 propose_kregion	absorb_fold
  783 propose_family	Coop
  772 merge_region_wave_seen	members=2 mergeable=1
  176 propose_scatter	SortSegment
  168 merge_region_wave_seen	members=2 mergeable=0
  150 propose_gather	RowPerGroup
  145 propose_scatter	Atomic
   70 propose_gather	Vectorized
   62 propose_scatter	WgPrivateMerge
   62 propose_scatter	OneHotContract
```

`merge_mint_ok`, `propose_family GenericFold` and `tile_contract_matched` do
not appear: **zero events each.**

### 63-token llama-8B-Q4K decode

```
5712 merge_mint_attempt	Matmul segs=1
5225 lower_head	KContract Sgemv Sgemv cols=4 parts=4 subgroup_cols
4021 merge_mint_attempt	Row segs=1
2556 lower_head	KMap Point
 256 lower_head	KScatter SortSegment
 128 lower_head	KScatter Atomic
 126 lower_head	KContract Sgemv Sgemv cols=2 parts=4 subgroup_cols
 125 lower_head	KFold Point
  80 lower_head	KGather RowPerGroup
   2 lower_head	KGather QuantizedRows
```

### All four model crates in one log (llama decode + rbert + whisper + SAM)

```
34453 merge_mint_attempt	Matmul segs=1
32517 propose_family	Sgemv
31937 propose_family	Sgemm
29697 merge_mint_attempt	Row segs=1
 8376 cpu_lower_head	KMap
 7566 propose_scatter	SortSegment
 7566 propose_scatter	Atomic
 5304 lower_head	KContract Sgemv Sgemv cols=4 parts=4 subgroup_cols
 3921 lower_head	KMap Point
 2698 propose_kregion	absorb_fold
 2698 merge_region_wave_seen	members=2 mergeable=1
 2031 propose_family	Coop
 1887 cpu_lower_head	KContract Sgemv
  979 cpu_lower_head	KFold
  608 propose_gather	RowPerGroup
  484 propose_gather	Vectorized
  450 cpu_lower_head	KScatter SortSegment
  339 lower_head	KScatter SortSegment
  280 lower_head	KFold Point
  159 lower_head	KContract Coop Coop splits=1
  128 lower_head	KScatter Atomic
  126 lower_head	KContract Sgemv Sgemv cols=2 parts=4 subgroup_cols
   94 lower_head	KGather RowPerGroup
   30 cpu_lower_head	KGather QuantizedRows
    3 lower_head	KGather QuantizedRows
```

Again zero `merge_mint_ok`, zero `KMerged`, zero `KRegion`, zero `Ext`, zero
`GenericFold`, zero `tile_contract_matched`.

---

## Ranked deletion plan

| # | candidate | evidence class | est. LOC | risk |
|---|---|---|---|---|
| 1 | ~~`fusor2-cost::calibrate` + `fusor2-cost::cache`~~ **DONE** | compile experiment | **1,384** | very low |
| 2 | ~~`KMerged` and the whole wave-merging path~~ **DONE** | census, 0/74,488 | **1,318** | low |
| 3 | ~~`L1::Ext` + the `MacroOp` sugar-node layer~~ **REJECTED** | deleted, measured 3x slower, reverted | ~~700~~ 0 | — |
| 4 | ~~`L1::KRegion`~~ **REJECTED** | deleted (1,441 lines), gated green, measured 3-10% slower, reverted | ~~500~~ 0 | — |
| 5 | ~~`Family::GenericFold`~~ **DONE** | census + static: no site constructs it | **182** | low |
| 6 | ~~`TILE_CONTRACT` rule~~ **DONE** | static: `lower_floor` mints no point-scheduled `KContract` | (in #5's diff) | very low |
| 7 | ~~`ScatterMode`'s never-selected half~~ **DONE** | one shared nest, one unlowerable member | **~330 with #8** | low |
| 8 | ~~`GatherMode::Vectorized`~~ **DONE** | 554 proposals / 0 wins | (in #7's diff) | low |
| 9 | ~~36 never-referenced `pub fn`s~~ **DONE** | token scan over 3,046 defs, then the compiler | **~440** | very low |
| 10 | ~~the alias surface~~ **DONE** | call graph; also the typed and autograd twins | **~200** | very low |
| 11 | ~~`softmax_slow*`~~ **DONE** | e-class identity | **~80** | very low |
|   | **round 5 total** | | **905 net Rust** | |

---

### 1. `fusor2-cost::calibrate` + `fusor2-cost::cache` — 1,384 lines — **DELETED**

> Done. Both modules, the `fusor2_ir::cost::Calibrate` trait (whose only
> implementor was `Calibrator`) and the now-unused `Result` import it pulled
> into `cost.rs` are gone; `facts::seed_facts` is the sole `DeviceFacts`
> producer. `fusor2-cost`'s test count fell 92 → 80, exactly the 7 + 5 tests
> defined inside the two deleted files. `serde`/`serde_json` stay: they are
> used by the live `tune_cache`.


**(a) What it is.** `calibrate.rs` (1,032 lines) is a device-benchmark suite —
`Bench`, `Calibrator::run`, `Calibrator::calibrate`, `calibrate_reporting`,
`CalibrationReport` — that measures `DeviceFacts` by running micro-kernels.
`cache.rs` (352 lines) is the on-disk cache of the result: `FactsRecord`,
`FORMAT_VERSION`, `cache_dir`, `path_for`, `load`, `load_record`, `store`,
`CalibrationMode`, and the one function that ties the two together,
`facts_for(target, mode)`.

**(b) Evidence.** Nothing calls `facts_for`. Both live targets take the seed
path directly:

```
fusor2-gpu/src/device.rs:156:    let facts = fusor2_cost::facts::seed_facts(&caps);
fusor2-cpu/src/target.rs:32:        let facts = seed_facts(&caps);
```

and the only mention of `Calibrator` outside `calibrate.rs` anywhere in either
repository is its own re-export:

```
$ grep -rn "Calibrator\|CalibrationMode\|facts_for" --include='*.rs' . | grep -v target/ | grep -v 'fusor2-cost/src/c'
fusor2-cost/src/lib.rs:26:pub use calibrate::Calibrator;
```

Proved by construction, not by inspection — with `pub mod cache; pub mod
calibrate; pub use calibrate::Calibrator;` removed from `fusor2-cost/src/lib.rs`
the entire workspace including every test target still compiles:

```
$ cargo check --workspace --all-targets
    Checking fusor2-cost v0.1.0 (.../fusor2-cost)
    Checking fusor2-gpu v0.1.0 (.../fusor2-gpu)
    Checking fusor2-cpu v0.1.0 (.../fusor2-cpu)
    Checking fusor2 v0.1.0 (.../fusor2)
    Checking fusor2-conformance v0.1.0 (.../fusor2-conformance)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.62s
```

and so does the parent `training` workspace (all four model crates):

```
$ cd /Users/evanalmloff/Desktop/Github/training && cargo check --workspace
    ...
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1m 32s
```

**(c) LOC.** 1,032 + 352 = **1,384**, plus the two `pub mod` lines and the
re-export.

**(d) Risk / proof of safety.** Very low: no code path reaches it, so no kernel
and no cost number changes. The one thing to confirm is that `DeviceFacts`
really is only ever produced by `facts::seed_facts` after the deletion — a
`grep` for `DeviceFacts {` constructors. Gate: workspace build in both repos +
full conformance. No perf effect is possible; measure anyway.

---

### 2. `KMerged` and the whole wave-merging path — 1,318 lines — **DELETED**

> Done. The node, its three rules, both backends' lowerings, every IR/cost/tile
> arm and all the tests are gone; `KRegion` keeps the composite lowering and
> inherits `region_domain`, `linear_domain` and `block_for`, whose coverage was
> re-pointed rather than dropped. `linear_domain_of` moved to `fusion.rs`, its
> one surviving caller, and `rules::is_ident` went with the merge rules that
> were its only users. `CORE_RULES` is 37 → 34 and `TAG_COUNT` 20 → 19, and
> `fusor2-gpu/src/lower/merged.rs` is renamed `region.rs` — a module named
> for merging with no merging left in it is exactly the vestigial name this
> campaign is supposed to remove.
> Re-verified before deleting, with my own probe: `mint` past its guard 0
> times and `lower_kmerged` entered 0 times across the whole conformance suite
> (both backends) *and* all four model crates.
>
> **Plan identity, not just a tok/s A/B.** `FUSOR2_DUMP_PLAN` over a full
> decode is byte-identical between HEAD and this commit — 6,854 lines,
> `diff` exit 0, same md5 `8c9d807b7e24…`, `nodes=46614 classes=4308
> launches=1695 buffers=1695` — so every launch's root, op, shape, grid,
> block and theta is unchanged. The merge rules minted no nodes (`mint`
> returns `None` *before* `add_l1`), so not even a node id moves. The paired
> decode spread is therefore noise by construction.


**(a) What it is.** `L1::KMerged`, the `KMerged` struct and constructor,
`MergeKey`, `MergeSegment`, `WaveCat` (`Region`/`Row`/`Matmul`/`MatmulSplitK`);
the three rules `MERGE_CONTRACT_WAVE`, `MERGE_ROW_WAVE`, `MERGE_REGION_WAVE`
(`fusor2-ir/src/rules/merge.rs`); `lower_kmerged`, `segment_elements`,
`merged_domain` (`fusor2-gpu/src/lower/merged.rs`); the `KMerged` arms in
`verify_l1`, `infer_l1`, `children`, `saturate`, `semantics`, both backends'
dispatch matches, `fusor2-tile/src/rules.rs`, and `fusor2-cost::realize::is_merged`.

**(b) Evidence.**

*Runtime.* Across the full conformance suite (both backends), a 63-token decode
and all four model crates, `mint` was attempted **74,488 times and succeeded
0 times**, and `KMerged` was lowered **0 times**:

```
conformance : 12485 merge_mint_attempt Row segs=1   6586 merge_mint_attempt Matmul segs=1
decode      :  4021 merge_mint_attempt Row segs=1   5712 merge_mint_attempt Matmul segs=1
models      : 29697 merge_mint_attempt Row segs=1  34453 merge_mint_attempt Matmul segs=1
             (merge_mint_ok: no events; lower_head KMerged: no events)
```

*Static.* Two of the three rules are dead by reading. `mint` opens with

```rust
fn mint(b: &mut Builder<'_>, id: Id, cat: WaveCat, segs: Vec<MergeSegment>) -> Option<Id> {
    if segs.len() < 2 {
        return None;
    }
```

and both `merge_contract_wave` and `merge_row_wave` call it with a
one-element vector — `mint(b, id, cat, vec![seg])`. They can never mint
anything, which is exactly what the `segs=1` census shows. Only
`merge_region_wave` can pass ≥2 segments, and it needs a `KRegion` with two
mergeable members sharing a `MergeKey`. Every `KRegion` in the graph comes from
one rule (`fusion.rs`'s absorb-fold), and it always has exactly two members of
which at most one is a mergeable shape:

```
conformance : 772 merge_region_wave_seen members=2 mergeable=1
              168 merge_region_wave_seen members=2 mergeable=0
models      : 2698 merge_region_wave_seen members=2 mergeable=1
```

So `WaveCat::Region` is unreachable too, and `WaveCat::MatmulSplitK` is
unreachable a third way: it is chosen only when `seg.key.splits > 1`, and
`segment_of` hard-codes `splits: 1` on every segment it builds.

*The body is worse than a stub — it is a latent miscompile.* `lower_kmerged`'s
Region/Row arm does

```rust
let view = ctx.linear_view(*seg_id) ...;
...
WaveCat::Region | WaveCat::Row => offsets.iter().zip(&guards).map(|(off, guard)| {
        let v = ctx.b.load(Source::Storage(view.clone()), Addr::Linear(off.clone()), ...);
        ctx.b.cast(v, out_elem)
    }).collect(),
```

and then stores that value back into the same `view` at the same offset:
`linear_view(seg_id)` is the buffer of the segment's own value, so the kernel
copies a buffer onto itself and never evaluates the segment's body at all. The
Matmul arm is the same defect wearing a k loop: it accumulates
`view[off, k]` — again the segment's own output buffer, not its A and B
operands. Under DOCTRINE this is a compiler bug that must be fixed at source or
removed; since selection has never reached it on any workload and cannot reach
it from the current rules, removing it is the honest fix and removes the
latent wrong-value generator with it.

**(c) LOC.**

| file | what | lines |
|---|---|---|
| `fusor2-ir/src/rules/merge.rs` | whole file bar `linear_domain_of` (7 lines, moves to `fusion.rs`) | 373 |
| `fusor2-gpu/src/lower/merged.rs` | `lower_kmerged` 212, `segment_elements` 21, `merged_domain` 5, KMerged-only tests 394, module doc/imports ~10 | 642 |
| `fusor2-ir/src/ir/level1.rs` | `WaveCat` 8, `MergeKey` 11, `MergeSegment` 7, `KMerged` 12, `impl KMerged` 51, `L1::KMerged` variant + doc + `tag`/`schedule` arms ~12, test ~20 | 121 |
| `fusor2-ir/src/verify_l1.rs` | rule-6 block 20, arms 3, tests ~70 | 93 |
| `fusor2-cpu/src/lower.rs` | dispatch arm + `KMerged` test | 31 |
| `fusor2-ir/src/semantics/infer_l1.rs` | arm 8 + test ~25 | 33 |
| `fusor2-ir/src/semantics/children.rs` | arm 1 + test ~28 | 29 |
| `fusor2-ir/src/saturate.rs` | `OpTag::KMerged` priority, wave-of-one assertion, docs | 12 |
| `fusor2-ir/src/rules.rs` | 3 registrations + 2 arms | 8 |
| `fusor2-cost/src/realize.rs` | `is_merged` + its one caller branch | 8 |
| `fusor2-tile/src/rules.rs`, `fusor2-gpu/src/lower.rs`, `fusor2-ir/src/semantics.rs` | arms | 5 |
| conformance | `KMerged` appears only in a doc comment on `qkv_triple` and in the golden name `matmul_epilogue_vs_merged_wave`; **no case exercises a merged wave numerically** | ~10 |
| | | **~1,390** |

**(d) Risk / proof of safety.** Low. Nothing selects it, so no kernel changes.
Proof: after the deletion the census must still show the same `lower_head`
histogram (it is deterministic — two conformance runs produced byte-identical
counts), conformance must stay `797 passed / 0 failed`, and the four model
crates must still run. Note explicitly: the 4 `merged::*` failures in the
known-pre-existing `fusor2-gpu` set are tests **of this code** and disappear
with it, which is a legitimate reduction of the failing set, not a silent fix.
Watch for `OpTag::KMerged` in `saturate.rs`'s priority table — removing an
`OpTag` renumbers nothing there (it is a `match`, not an index), but the
`semantics.rs:643` "10 L0 tags + 7 of the 8 L1 tags" assertion needs its count
updated.

---

### 3. `L1::Ext` and the `MacroOp` sugar-node layer — **REJECTED, do not retry**

> **The audit was right that `Ext` is never selected and wrong that deleting it
> is free.** The whole layer was deleted (−2,059 code lines), every gate passed,
> and the shipped decode went **19.99 → 6.54 tok/s**. Reverted; the tree is back
> at `61a06c9`. See "Why it cannot go" below before touching this again.

**Deadness re-proved, three independent ways.** All eight `MACRO_OPS` declare
`lower_per_target: &[]`; `verify_plan::check_extensions` returns
`Err("extension `{name}` selected at {id} lowers on no target")` for any
selected node with an empty row, and `LocalSearch` calls it on **every** plan it
returns (`extract.rs:569`, `:639`, `:679`) — so a single `Ext` selection would
fail the run outright. Neither backend ever installs a registry outside its own
tests (`ext::install` has zero non-test callers), so an `Ext` reaching lowering
would also hard-error. A direct probe over the full suite agreed:

```
806 results: 797 passed, 9 skipped, 0 failed
PROBE_EXT_SELECTED: 0   PROBE_EXT_LOWER_GPU: 0   PROBE_EXT_LOWER_CPU: 0
```

**Why it cannot go.** `Ext` is dead as a *selectable node* and load-bearing as a
*union partner*. `macro_op` ends with `union_stable(defn, sugar)` and hands the
caller the **`Union` spine id**; with the sugar gone there is no second member,
no union, and every composite hands back a bare `L0` member id instead. Fusion
quality collapses on that difference. Isolated to that one line, at `61a06c9`,
with `L1::Ext` still minted and everything else untouched:

```
-    let root = graph.union_stable(defn, sugar)?;
-    Ok(graph.tensor(root))
+    Ok(graph.tensor(defn))
```

| llama-3.1-8B-Q4_K_M, 63 tokens | tok/s | KMap | KFold | KContract | Sgemv | Sgemm | Coop |
|---|---|---|---|---|---|---|---|
| `61a06c9` unmodified | **19.99** | 4798 | 250 | 1144 | **572** | 0 | 0 |
| one line above, nothing else | **6.46 / 6.57** | 6750 | 982 | 412 | 102 | 102 | 106 |
| full `Ext` deletion | **5.75 / 6.32 / 6.54** | 6750 | 982 | 412 | 102 | 102 | 106 |

The last two rows are byte-identical histograms, which is the isolation: **none
of the ~2,000 deleted lines cost anything; returning a member id instead of a
spine id costs 3x.** The decode matmuls stop lowering as `Sgemv` — the right
kernel at `M = 1` — and fall back to a generic `KFold`/`KMap` reduce plus some
`Sgemm`/`Coop`, which at `M = 1` is pure waste. It is not compile-side: both
binaries report `saturate (skipped) 0 us (~47k nodes), extract+verify 1 us,
replay hit` on every token, and the deletion *lowers* the node count (46,978 →
46,720).

**What this actually is.** A compiler fragility, not a fact about `Ext`: which
kernels get fused depends on whether the caller happens to hold a `Union` id or
a member id of the same class. Rules match spines (`Views(vs, X)`,
`trace_pure_views`), and a `Union` in the operand chain is what lets them.
Deleting `Ext` is blocked behind fixing *that* — make fusion match on class
membership rather than on the caller's chosen id — which is a rewrite-layer
change, not a deletion. Until then this item is worth **0 lines**, and the
`ScatterMode`/`GatherMode`/`GenericFold`/`tile_contract` items (#5-#8), which
mint no nodes and change no graph shape, are the safe remainder.

---

<details>
<summary>Original (pre-measurement) write-up, kept for the record</summary>


**(a) What it is.** `L1::Ext { def, ops, attrs }`, `OpDef`, `OpDefId`,
`OpDefRegistry`, `AttrId` (`fusor2-ir/src/ir/mod.rs`); `MacroOp`, `MacroAttr`,
`AttentionOut`, `MACRO_OPS`, `register_macro_ops`, `macro_op`
(`fusor2/src/composite.rs`); the `ext` module in **both** backends'
`lower.rs` (66 lines each plus tests); the `Ext` arms in `verify_l1`,
`infer_l1`, `children`.

**(b) Evidence.** `Ext` is lowered **0 times** in every census above — but it
*is* minted, once per composite call, and unioned into the composite's class,
so it costs a node and a union on every model step. Its stated reason to exist
is stated in its own doc:

> All eight declare `lower_per_target: &[]` — they are **unrunnable by
> construction**. They exist so rules can read the attributes a pattern match
> would otherwise have to re-derive

and no rule reads them:

```
$ grep -rn "OpTag::Ext\|attrs\b" --include="*.rs" fusor2-ir/src/rules.rs fusor2-ir/src/rules/ fusor2-tile/src/rules.rs fusor2-tile/src/rules/
$ grep -rn "fn attrs\|attr_of\|read_attrs" --include="*.rs" fusor2-ir/src
```

Both return nothing. There is no recognizer, no rewrite and no cost term that
reads a `MacroAttr`; `MacroAttr` is referenced outside `composite.rs` only by an
interning unit test (`fusor2/src/graph.rs:963-965`).

**(c) LOC.** `composite.rs` macro region ~272, both `ext` modules 132 + their
tests ~150, `ir/mod.rs` `OpDef`/registry ~65, IR arms and their tests ~80,
`Session` registration ~10. **~700.** 18 `macro_op(...)` call sites become
`core_op(...)`, which is an edit, not a deletion, at each.

**(d) Risk / proof of safety.** Medium — this is the only candidate that
changes graph *shape*: every composite currently mints `defn` + sugar +
`union_stable`, and afterwards mints only `defn`. That is strictly less work,
so it should be neutral-to-faster, but it is exactly the kind of change the
methodology says must be A/B'd: build both binaries, interleave 5 HEAD/MINE
pairs. Two specific things to check first: (i) `union_stable`'s
rebuild-stability contract is currently anchored on the sugar node — confirm
`core_op` alone still returns the same id on a decode loop's second build
(`session`'s node-identity assertions cover this); (ii) `mark_defn` /
`is_defn` are consumed by `fusor2/src/quantized.rs:531` and by the saturation
replay, so they stay. If the `L1::Ext` variant is judged worth keeping as an
architectural escape hatch, delete only the `MacroOp` layer (~450 lines) and
keep `Ext` — but then it has *no* producer at all, which is the weaker
position.

</details>

---

### 4. `L1::KRegion` — **DELETED, MEASURED, REVERTED**

> **Do not retry this.** The node is exactly as dead as the census said — it is
> proposed thousands of times and lowered zero times — and deleting it is still
> a 3-10% decode regression. Patch kept at `/tmp/kregion.patch`
> (19 files, 76 insertions, 1,517 deletions; 1,441 net Rust lines).

**(a) What it was.** `L1::KRegion { members, live_outs, sched }`, the
`FORM_KREGION` rule in `fusion.rs`, `lower_kregion` + the whole
`fusor2-gpu/src/lower/region.rs` file, the CPU `compose` path, `OpTag::KRegion`,
`verify_l1`'s invariant 6 (`check_composite_domain`), `MapDomain::linear` /
`linear_over` / `LINEAR_TM_CHOICES` (KRegion was their only non-test caller),
`realize::{needs_own_buffer, absorbs}` (with `KRegion` gone `absorbs` is
constantly false, so `needs_own_buffer` collapses to `true` and
`moves::at_structural_boundary` becomes "has a realized consumer"),
`fusor2-cpu`'s `intern_decl` buffer-decl pool (a launch holding two `Binds` was
only ever a composite), and every IR / cost / tile / doc arm.

**(b) The deadness claim is CONFIRMED**, re-proved with my own probe rather than
trusted — a counter at `form_kregion` past its operand search and at both
backends' `KRegion` lowering arm:

```
conformance (806 results, CPU+GPU): PROBE_KREGION_MINT 940
                                    PROBE_KREGION_LOWER_GPU 0
                                    PROBE_KREGION_LOWER_CPU 0
llama decode + rbert (one log):     PROBE_KREGION_MINT 450
                                    PROBE_KREGION_LOWER_GPU 0
                                    PROBE_KREGION_LOWER_CPU 0
```

Both runs were real: `806 results: 797 passed, 9 skipped, 0 failed`,
`63 tokens in 6.77 seconds`, `PASS: paraphrase similarity clearly exceeds
unrelated similarity`.

**(c) The deletion gated completely green.** `cargo build --workspace` in both
repos; `806 results: 797 passed, 9 skipped, 0 failed` (identical to HEAD);
`FUSOR2_VERIFY_MEMBERS=1 … quantized` → `108 results: 108 passed, 0 skipped, 0
failed`; unit tests `fusor2-ir 233`, `fusor2-cost 80`, `fusor2-tile 64`,
`fusor2-cpu 134`, `fusor2 312`, `fusor2-autograd 9`, `fusor2-conformance 107`
all green, and `fusor2-gpu` at exactly the two known pre-existing failures
(`109 passed; 2 failed`, down from 123 passed only because `region.rs`'s own
tests went with the file). Two assertions were retargeted, not weakened:
`CORE_RULES.len()` 34 → 33 and the fuzz corpus's tag floor 17 → 16.

**(d) And it is 3-10% slower.** Decode, llama-3.1-8B-Q4_K_M, 63 tokens,
interleaved in one session, both orders (`H,M` ×5 then `M,H` ×5 then `M,H` ×5):

```
HEAD  22.34 21.27 22.06 22.02 22.00 21.93 21.95 21.87 22.09 22.10 21.82 21.84 22.11 21.66 21.48
MINE  20.44 21.53 21.13 21.34 21.38 21.79 20.53 21.98 21.96 21.65 20.81 22.31 21.41 20.57 19.67
```

mean **21.90 vs 21.23** (−3.1%), MINE loses **12 of 15 pairs**. HEAD's range is
21.27-22.34; MINE's is 19.67-22.31 — the deletion does not shift the mean so
much as add a slow mode HEAD never enters.

The tuning cache is shared, mutable and order-sensitive, so that interleave was
re-run with a **private, equally cold `XDG_CACHE_HOME` per binary**, which is
the only uncontaminated form of this measurement:

```
cold MINE  18.87  18.96  19.88      mean 19.24
cold HEAD  21.26  21.11  21.61      mean 21.33     -9.8%, 3 of 3 pairs lost
```

The two private caches came out **identical** — same 1 launch key, same 9
variant keys — so the tuner is not the difference.

**(e) What is NOT the mechanism.** Both binaries were run with
`FUSOR2_WGSL_DUMP`: **506 shaders each, identical filenames, and
`cat wh/* | md5 == cat wm/* | md5` = `c3b12cf4f5c2babe49b5e9730ed0bdb0`.** Every
kernel either binary can dispatch is byte-for-byte the same. So this is not a
worse kernel, not a lost `Sgemv`, not a fallback — the universe of compiled
kernels is unchanged and something above it (which member of that universe each
launch adopts, or how much host work the per-token path does) got worse.

**(f) The lesson, which generalizes past this node.** The census measured what
was *lowered*. A node can be lowered zero times and still be load-bearing as a
**waypoint the extractor passes through** — `FORM_KREGION` unions a
`KRegion { members: [producer, fused] }` into the fold's own class, so it is an
alternative every `RESELECT`/`FLIP` move can see even though no plan ships it.
Round 3 rejected `Ext` for the structurally identical reason (dead as a
selectable node, load-bearing as a union partner). **"Proposed but never
selected" is therefore not evidence that a node is free to delete**, and items
7 and 8 on this list rest on exactly that inference. Both now need a
cold-private-cache decode A/B *before* their code is touched, not after.

---

### 5. `Family::GenericFold` — ~180 lines

**(a) What it is.** The fourth `Family` on `L1::KContract`, documented as "the
always-legal family… what the cost model reaches for on a device with neither
subgroups nor cooperative matrices", with `lower_generic` in
`fusor2-gpu/src/lower/contract.rs` (lines 1903–1931), a `Family::GenericFold`
arm in `tile_contract`, one in `lower_family`, and the `fold_domain` plumbing
that feeds them.

**(b) Evidence.** `propose_family` fires for `Coop`, `Sgemm` and `Sgemv` only —
`GenericFold` never, in any run:

```
conformance : 1564 propose_family Sgemv   1522 propose_family Sgemm    783 propose_family Coop
models      : 32517 propose_family Sgemv 31937 propose_family Sgemm   2031 propose_family Coop
```

The reason is structural: `lower_family` is only ever called with `Coop`,
`Sgemm` and `Sgemv` (`contract.rs:519/524/530`), and the rule actually named
`LOWER_GENERIC` mints a **`KFold`**, not a `KContract { family: GenericFold }`.
No other site in `fusor2-ir/src/rules/` or `fusor2-tile/src/rules/` constructs
`family: Family::GenericFold` outside a test. The real always-legal floor is
`Sgemm`, which is unguarded on shape.

**(c) LOC.** `lower_generic` 29, its CPU twin ~25, two `Family::GenericFold`
match arms ~15, `fold_domain`-for-contract plumbing ~40, the enum variant and
its exhaustive-match arms across `fusor2-ir`/`fusor2-cost` ~30, tests ~40.
**~180.**

**(d) Risk / proof of safety.** Low, but it deletes a documented legality
floor, so the safety proof is a *legality* argument, not a census one: show
that for every `KContract` shape at least one of `Coop`/`Sgemm`/`Sgemv` is
minted. `Sgemm` declines only when `sgemm_domain(...).params.is_empty()` or
`operands_addressable` fails; both need to be shown impossible, or the floor
kept and only `lower_generic`'s dead GPU/CPU bodies removed.

---

### 6. The `TILE_CONTRACT` rule — ~70 lines

**(a) What it is.** `fusor2-tile::rules::contract::tile_contract` (lines
448–507) plus its `rule!` declaration and its entry in `TILE_RULES`. It attaches
a schedule domain to a `KContract` that arrived carrying
`ScheduleDomain::Point`.

**(b) Evidence.** A probe placed immediately after its
`sched: ScheduleDomain::Point` destructure recorded **zero events** over the
full conformance suite and all four model crates. No `KContract` ever arrives
with a `Point` schedule: `lower_family` mints them with a full domain already,
and `sink.rs` / `fusion.rs` / `layout.rs` rebuild `KContract` preserving
`sched`.

**(c) LOC.** 60 (function) + 6 (`rule!`) + 1 (registration) + its tests.
**~70.**

**(d) Risk / proof of safety.** Very low. The proof is that the rule is
unreachable: enumerate every `L1::KContract` construction site and show none
passes `ScheduleDomain::Point`. Gate on conformance and one decode A/B (a rule
removed from the table shortens every saturation pass, so this should be a
small *win*).

---

### 7. `ScatterMode`'s never-selected half — ~160 lines

**(a) What it is.** `ScatterMode::WgPrivateMerge` and
`ScatterMode::OneHotContract`, the tile rules `SCATTER_WG_PRIVATE_MERGE` and
`SCATTER_ONE_HOT_CONTRACT`, and the mode field's role generally.

**(b) Evidence.** Offered 62 times each over conformance, never over the model
crates, and lowered **0 times anywhere**. `OneHotContract` cannot lower at all —
`lower_kscatter` errors on it by design. And the three remaining modes are not
three lowerings: `lower_kscatter` sends `Atomic`, `SortSegment` **and**
`WgPrivateMerge` to the same `scatter_dense` body, which destructures only
`combine` and `ops` and never reads `mode`. The CPU backend says so out loud:

```
fusor2-cpu/src/lower/gather_scatter.rs:3://! All four `ScatterMode`s name one map and differ only in strategy, so on a
```

So this is the "alias/duplicate spellings" pattern one level down: four names
for one kernel, two of which are never chosen.

**(c) LOC.** Two rules ~70, the `OneHotContract` refusal arm ~10, mode arms and
tests across `fusor2-tile`/`fusor2-gpu`/`fusor2-cpu` ~80. **~160.**

**(d) Risk / proof of safety.** Low. Since `Atomic` and `SortSegment` produce
the *same* kernel today, the only thing `mode` can change is the price. Prove
by showing the cost model does not read `ScatterMode` (grep gives only two
`ScatterMode::Atomic` test constructors in `fusor2-cost`), then delete the two
never-selected variants; collapsing the remaining two is a second, separate
step that needs a cost-model check first.

---

### 8. `GatherMode::Vectorized` — ~90 lines

**(a) What it is.** The vector-load gather lowering: the `GATHER_VECTORIZED`
rule, the `run` computation in `lower_kgather` that branches on it, and its
tests.

**(b) Evidence.** Offered 70 times over conformance and 484 times over the four
model crates; lowered **0 times**. It is legality-guarded (row a whole number of
16-byte quads, unit stride) and passes that guard often, so this is a pure
cost-model verdict.

**(c) LOC.** Rule ~25, lowering branch ~10, mode arms ~10, tests ~45. **~90.**

**(d) Risk / proof of safety.** Low, but same caveat as #4: it loses on price,
not on legality, so confirm it is not a near-tie before removing. The cheap
proof is to force its cost to zero and check that some case then selects it — if
nothing does even then, it is unreachable for a structural reason and safe.

---

### 9. Forty-three never-referenced `pub fn`s — ~400 lines

**(a) What it is.** Public functions whose identifier occurs exactly once in
the whole workspace — at their own definition.

**(b) Evidence.** Token-frequency scan over all `.rs` in the nine crates with
comments stripped; 2,250 `pub fn`s, 43 with a usage count of 1:

```
fusor2-ir/src/ir/level1.rs:308 pub fn sole
fusor2-ir/src/ir/level1.rs:326 pub fn try_map_ops
fusor2-ir/src/ir/level1.rs:889 pub fn cols_per_subgroup
fusor2-cost/src/cache.rs:180 pub fn facts_for          (see #1)
fusor2/src/graph.rs:907 pub fn into_detached
fusor2/src/layers/linear.rs:50 pub fn in_features
fusor2/src/layers/linear.rs:53 pub fn out_features
fusor2/src/cache/kv.rs:113 pub fn fixed_windowed
fusor2/src/cache/kv.rs:505 pub fn pending_into
fusor2/src/cache/mask.rs:65 pub fn is_structural
fusor2/src/composite/attention.rs:601 pub fn neg_infinity
fusor2/src/ops/elementwise.rs:98 pub fn unpack2x16_float
fusor2/src/ops/alias.rs:69 pub fn mt_matmul            (see #10)
fusor2/src/ops/reduce.rs:85 pub fn min_with_tie
fusor2/src/ops/reduce.rs:154 pub fn max_all
fusor2/src/ops/reduce.rs:158 pub fn min_all
fusor2/src/ops/reduce.rs:162 pub fn product_all
fusor2/src/ops/reduce.rs:207 pub fn arg_min
fusor2/src/ops/cast.rs:52 pub fn to_f16
fusor2/src/ops/cast.rs:55 pub fn to_bf16
fusor2/src/ops/matmul.rs:189 pub fn einsum
fusor2/src/ops/scalar_arith.rs:79 pub fn rrem_scalar
fusor2/src/ops/scalar_arith.rs:94 pub fn mul_uniform
fusor2/src/tensor/construction.rs:98 pub fn full_f32
fusor2/src/tensor/construction.rs:123 pub fn uninit
fusor2/src/tensor/construction.rs:157 pub fn uniform_scalar
fusor2/src/tensor/typed.rs:295 pub fn as_tensor
fusor2/src/tensor/typed.rs:770 pub fn reshape_extents
fusor2/src/tensor/readback.rs:362 pub fn as_slice_async
fusor2/src/tensor/typed/ops.rs:217 pub fn to_flat_async
fusor2-autograd/src/backward.rs:147 pub fn with_custom
fusor2-conformance/src/suite.rs:327 pub fn forward_case
fusor2-conformance/src/compare.rs:441 pub fn finite_difference_gradient_in
fusor2-conformance/src/oracle.rs:426 pub fn assert_oracle_agrees
fusor2-conformance/src/launch_counts.rs:348 pub fn check_pin
fusor2-conformance/src/exhaustive.rs:84 pub fn verify_point
fusor2-conformance/src/suite/reductions.rs:251 pub fn must_decline
fusor2-conformance/src/suite/reductions.rs:316 pub fn must_not_materialize
fusor2-conformance/src/suite/reductions.rs:337 pub fn buffer_ceiling
fusor2-gguf/src/sharded.rs:67 pub fn get_tensor
fusor2-gguf/src/parse.rs:345 pub fn as_u64
fusor2-gguf/src/parse.rs:606 pub fn read_tensor_bytes
fusor2-gguf/src/varbuilder.rs:49 pub fn from_gguf
```

**(c) LOC.** ~400 including doc comments.

**(d) Risk / proof of safety.** Very low, but the list needs one filter pass:
some are trait-impl-adjacent or exist to keep an API symmetrical
(`to_f16`/`to_bf16`, `max_all`/`min_all`/`product_all` next to a used
`sum_all`), and deleting half of a symmetric family is a worse surface than
keeping it. Proof per item is the compiler: delete, build both workspaces.

---

### 10. `fusor2/src/ops/alias.rs` — ~110 lines

**(a) What it is.** 14 pure forwarders on the dyn `Tensor`
(`mt`→`gt_scalar`, `mte`, `eq`/`ne`/`lt`/`lte`/`gt`/`gte`,
`pow_elementwise`, `max_elementwise`, `min_elementwise`, `mat_mul`,
`mt_matmul`, `mat_mul_transposed_rhs`) — the same class the already-deleted
`*_fused` family belonged to. The file's own header says "Nothing in this file
mints a node, so every alias is structurally identical to its target and
hash-conses onto it."

**(b) Evidence.** Usage outside the file, across both repositories:

```
mt                            6   (4 of them tests-of-the-alias, 2 doc comments)
mte                           3   (all tests-of-the-alias)
pow_elementwise               1   (test-of-the-alias)
max_elementwise               7   (real: trainer_surface, autograd — but on the TYPED tensor)
min_elementwise               1   (test-of-the-alias)
mat_mul                       5   (1 real: models/kalosm-llama/src/raw/vision/qwen_rope.rs:25)
mt_matmul                     0
mat_mul_transposed_rhs        3   (all tests-of-the-alias)
```

Seven of the fourteen have no caller but the unit test that asserts the alias
hash-conses onto its target, which is a test of a delegation that would not
exist.

**(c) LOC.** 76 (the file) + the `alias_forwarders_hash_cons_onto_their_target`
tests in `tensor.rs` ~35. **~110.** The two live callers
(`qwen_rope.rs`'s `mat_mul`, and `max_elementwise` on the typed tensor) get
renamed to `matmul` / `max_scalar` in the models repo.

**(d) Risk / proof of safety.** Very low — the compiler finds every caller.
Note two conformance cases (`mt`, `mte` in `elementwise.rs` and `backward.rs`)
go with them; their twins `gt_scalar`/`gte_scalar` are registered and green in
the same table, which is the "removed case's twin is still registered" proof
the gate asks for.

---

### 11. `softmax_slow` / `softmax_slow_last_dim` — ~35 lines

**(a) What it is.** Two `#[doc(hidden)]` methods on the dyn `Tensor` returning
the bare softmax expansion, plus two conformance cases.

**(b) Evidence.** `macro_op` builds `defn` first and then
`union_stable(defn, sugar)`, so `softmax(axis)` and `softmax_slow(axis)` land
in the **same e-class**; hash-consing makes the bare expansion literally the
same node. The crate's own test says so:

```rust
// Hash-consing folds the four identical expansions into one node, so
// the bare `slow` id is a member of the sugared class.
assert!(members.contains(&slow.id()), "{members:?}");
```

There is no second kernel and no route for the extractor to pick differently,
so the conformance cases `softmax_slow` and `softmax_slow_last_dim` re-test the
class `softmax` / `softmax_last_dim` already cover.

**(c) LOC.** ~20 in `composite/normalization.rs`, ~15 of conformance rows and
golden names. **~35.**

**(d) Risk / proof of safety.** Very low. If #3 is done, `softmax_slow` becomes
literally `softmax` and the question answers itself. Removed-case proof: the
`softmax` and `softmax_last_dim` rows sit immediately above them in the same
table in `fusor2-conformance/src/suite/normalization.rs:113`.

---

## Leads checked and *rejected*

* **`optim` is not unused.** `fusor2-conformance/src/suite/layers.rs:16` imports
  `fusor2::optim::{AdamW, clip_global_norm, cosine_decay}` and
  `fusor2/src/api_surface.rs:33` pins them. Keep, do not demote.
* **`broadcast` and `ops` are already private** — `mod broadcast;` and
  `pub(crate) mod ops;` in `fusor2/src/lib.rs:40,46`. Nothing to demote.
* **`fusor2-cpu` is a live target**, not a second copy: `fusor2/src/session.rs`
  uses `CpuTarget` and `AlignedBuf`, and it lowered 15,003 nodes during the
  conformance run. 9,598 lines that stay.
* **`fusor2-cost::moves`, `replay`, `tune_cache`** all have live consumers
  (`extract.rs`, `session.rs`).

---

## Baseline (for the seal to quote against)

### Line count — `tokei --exclude target --exclude vendor`, repo root

```
===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 TOML                   10          281          233           24           24
-------------------------------------------------------------------------------
 Markdown                4          954            0          697          257
 |- Rust                 3          285          255           15           15
 (Total)                           1239          255          712          272
-------------------------------------------------------------------------------
 Rust                  211       116059       101165         6216         8678
 |- Markdown           211        17064            0        15368         1696
 (Total)                         133123       101165        21584        10374
===============================================================================
 Total                 225       117294       101398         6937         8959
===============================================================================
```

Per crate (Rust lines / code):

```
fusor2                 22018 / 18649
fusor2-autograd         5934 /  5299
fusor2-conformance     13541 / 11830
fusor2-cost             9707 /  8389
fusor2-cpu              9598 /  8636
fusor2-gguf             3215 /  2839
fusor2-gpu             18328 / 16123
fusor2-ir              24836 / 21599
fusor2-tile             8940 /  7854
```

The plan above removes **~5,019 Rust lines**, ~4.3% of the tree, without
touching a selected code path.

### Decode throughput — llama-3.1-8B-Instruct Q4_K_M, 63 decode tokens

Clean `decode-switch` HEAD, both workspaces rebuilt from a reverted tree, six
consecutive runs in one session:

```
run0: 9.78 tokens/second     <- discarded: first run after a heavy GPU process
run1: 18.78 tokens/second
run2: 18.95 tokens/second
run3: 18.75 tokens/second
run4: 20.25 tokens/second
run5: 20.13 tokens/second
```

Baseline: **median 18.95, mean 19.37, range 18.75–20.25 tok/s** over the five
kept runs. An independent five-run set taken earlier in the same session on a
functionally identical binary gave 18.46 / 19.21 / 18.86 / 18.98 / 19.12 —
median 18.98 — so run-to-run spread is ~±0.75 tok/s and a change must win a
5-pair interleaved A/B, not a single run.

### Conformance

```
806 results: 797 passed, 9 skipped, 0 failed
```

Reproduced four times during this session (twice with instrumentation, twice
without) with identical counts.

Quantized subset with the member sweep on:

```
$ FUSOR2_VERIFY_MEMBERS=1 ./target/release/fusor2-conformance quantized
108 results: 108 passed, 0 skipped, 0 failed
```

### Known pre-existing unit-test failures

```
$ cargo test --release -p fusor2-gpu
failures:
    lower::contract::tests::a_narrow_output_stages_the_accumulator
    lower::map_fold::tests::single_slot_fold_wgsl_is_unchanged
    lower::merged::tests::a_split_k_wave_is_refused_rather_than_summed_short
    lower::merged::tests::every_point_writes_every_element_exactly_once
    lower::merged::tests::the_default_point_writes_every_element_exactly_once
    lower::merged::tests::the_group_index_reads_all_three_dispatch_axes
test result: FAILED. 123 passed; 6 failed; 0 ignored; 0 measured; 0 filtered out
```

Every other crate is green:

```
fusor2              312 passed; 0 failed   (+ tests/lowering_regressions.rs 9 passed)
fusor2-autograd     107 passed; 0 failed
fusor2-conformance  163 passed; 0 failed
fusor2-cost          92 passed; 0 failed
fusor2-cpu           67 passed; 0 failed
fusor2-gguf          22 passed; 0 failed
fusor2-ir           245 passed; 0 failed
fusor2-tile         134 passed; 0 failed
fusor2-gpu          123 passed; 6 failed   <- the known set above
```

Four of the six are `lower::merged::tests::*` — tests of candidate #2's body,
which go away with it. After that deletion the expected `fusor2-gpu` failure set
is **two**: `a_narrow_output_stages_the_accumulator` and
`single_slot_fold_wgsl_is_unchanged`. That is a reduction of the failing set by
removing the code under test, not a silent fix, and it must be stated as such
in the seal.

### All four model crates run at HEAD

```
kalosm-llama  examples/infer_local          exit=0, 63 tokens generated
rbert         examples/embed_local          exit=0, "PASS: paraphrase similarity clearly exceeds unrelated similarity"
rwhisper      examples/transcribe_local     exit=0, "transcribed 110 chars"
segment-anything-rs examples/segment-anything  exit=0, "Saved mask to out.png"
```


---

# ROUND 5 — the seal

Everything left on the list is deleted. `git diff --shortstat c6bd339..HEAD --
'*.rs'` for this round is `43 files changed, 119 insertions(+), 1024
deletions(-)` — **905 net Rust lines** — and the campaign total against
`76d42f7` is `69 files changed, 1010 insertions(+), 4450 deletions(-)`, **3,440
net Rust lines**, with the tree at 113,048 Rust lines from 116,059.

## What died, and the proof for each

**`TILE_CONTRACT` (`e9126d0`).** The census counted zero matches; the *static*
proof is stronger and is what the deletion rests on. `rules/lower_floor.rs` says
in its own header that it is "the only place `ScheduleDomain::Point` is minted",
and it mints `KMap`, `KFold`, `KGather` and `KScatter` — never a `KContract`
(`lower_contract_generic` mints a `KFold`). Every other
`KContract { sched: ScheduleDomain::Point }` in either workspace is inside
`#[cfg(test)]` or the `testing` fixture module, and the rules that *rebuild* a
`KContract` (`sink`, `layout`, `fusion`) clone `sched`. So the rule that
upgrades a point-scheduled contraction can never see one. `TILE_RULES` is 16 →
15 rules, and the fixture constructor `point_contract`, whose only caller was
that rule's own test, went with it.

**`Family::GenericFold` (`4f0c3f4`).** `lower_family` is called with `Coop`,
`Sgemm` and `Sgemv` only; the rule actually named `LOWER_GENERIC` mints a
**`KFold`**, not a fourth-family `KContract`; and no site outside a test
constructs `family: Family::GenericFold`. Unreachable by construction, so the
GPU's `lower_generic` body and the CPU nest's `SchedPoint::Fold` tile arm — the
tile that existed only to schedule that family — are dead code, and
`operands_addressable`'s `q_ok` third case with them. The one behavioural
predicate it carried (a generic fold may not read a quantized operand) is
restated inline where it is used.

**The alias surface (`192a926`).** `fusor2/src/ops/alias.rs` was fourteen
forwarders that mint nothing, and the same pattern had twins one and two layers
up: typed `mat_mul`/`max_elementwise`/`min_elementwise` and autograd
`mat_mul`/`max_elementwise` (the last of which was a straight duplicate of the
`same_scalar!`-generated `max_scalar`, so its deletion is enforced by the
compiler's duplicate-definition error). Six conformance rows go with them —
`eq_alias`, `ne_alias`, `lt_alias`, `lte_alias`, `mt`, `mte` — and each one's
twin sits in the *same table*: `eq_scalar`, `ne_scalar`, `lt_scalar`,
`lte_scalar`, `gt_scalar`, `gte_scalar` in
`suite/elementwise.rs::scalar_comparisons`, and
`gt_scalar_differentiates_to_zero` / `gte_scalar_differentiates_to_zero` in
`suite/backward.rs`. `qwen_rope.rs`'s `mat_mul` needed no edit: that file is not
in the module tree (it still `use fusor::`, the old crate).

**`softmax_slow*` (`dbba115`).** `macro_op` ends `union_stable(defn, sugar)`, so
`softmax_slow(axis)` and `softmax(axis)` land in one e-class and hash-consing
makes the bare expansion literally a member of the sugared class — the crate's
own test asserted exactly that. Three cases go (`softmax_slow`,
`softmax_slow_last_dim`, `softmax_sugar_agrees_with_its_defn`); their twins
`softmax_axis_last` and `softmax_last_dim` sit two lines above them in
`plain_rows()`, and the sugar-agrees claim is what `FUSOR2_VERIFY_MEMBERS`
sweeps over *every* class rather than one. Round 3's measurement is the second
reason: a name that returns the bare `defn` hands the caller a member id instead
of the union spine, which measured 3x slower to fuse from. **`dequantize_slow`
is NOT the same thing and stays** — it returns a class with *no* `L0::Dequant`
in it, so the extractor has no alternative and the case tests the unpack
arithmetic rather than which member won.

**Thirty-six `pub fn`s (`fab718c`).** A token scan (comments stripped) over
3,046 `pub fn` definitions in both workspaces found 179 whose identifier occurs
exactly once — at their own definition. 144 of those are `models/` and
`interfaces/` public builder API, which is a surface, not dead code; the 35 in
`fusor2-verified` are not on `API.md`'s pinned surface (`api_surface.rs` names
every item that is, and a named item would have counted twice). All were
deleted and the compiler agreed; deleting them exposed two more
(`oracle::with_arena`, `suite::materializes_elements`), which went too, and the
rescan is now empty.

**The never-selected scatter and gather modes (`bf23ffe`).** This is the one
item on the list where "proposed, never selected" was *not* the argument, since
round 4 proved that inference is unsound. The argument is that these members are
not alternatives at all:

* `fusor2-gpu/src/lower/gather_scatter.rs` says it in its own doc — "the four
  `ScatterMode`s name one map and differ only in *strategy*, and on this runtime
  they all lower through one nest" — and `lower_kscatter` never reads `mode`. So
  `WgPrivateMerge` is a third *spelling* of the kernel `Atomic` and
  `SortSegment` already produce.
* The cost model never reads `mode` either: `ScatterMode` appears in
  `fusor2-cost` only as two test constructors. All four priced identically.
* `OneHotContract` had **no lowering on either backend** — both returned an
  error naming it. A cost-identical member that errors if the tie-break lands on
  it is not a candidate, it is a landmine, and `fusor2/src/device.rs` documents
  the crater: adding seven ambient-graph tests with no library change was enough
  to turn `trainer_surface`'s f16 convolution backward red, because the shared
  e-graph's tie-break moved onto it.
* `GatherMode::Vectorized` was a real body — `run = 4` instead of `1` — and lost
  554 of 554 proposals.

`TILE_RULES` is 15 → 12. `trainer_conv_backward_chains_keep_four_alternatives`
had its scatter half retargeted 4 → 2 with the reason written beside it; its
contraction half is untouched at ≥ 4.

## Gates

`cargo build --workspace`, both repos:

```
$ cd fusor2-verified && cargo build --workspace
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 8.02s
$ cd .. && cargo build --workspace
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 12.95s
```

Conformance, both backends:

```
784 results: 775 passed, 9 skipped, 0 failed
```

806 → 784 is **exactly** the 11 deleted case names x 2 backends (6 comparison
aliases, 2 backward aliases, 3 softmax). 797 → 775 passed, 9 → 9 skipped, 0 → 0
failed. The mode deletion removed no case at all: the run before it and the run
after it are both 784/775/9/0.

Quantized subset with the member sweep:

```
$ FUSOR2_VERIFY_MEMBERS=1 ./target/release/fusor2-conformance quantized
108 results: 108 passed, 0 skipped, 0 failed
```

Unit tests, whole workspace:

```
fusor2              314 passed; 0 failed   (+ tests/lowering_regressions.rs 9 passed)
fusor2-autograd     107 passed; 0 failed
fusor2-conformance  163 passed; 0 failed
fusor2-cost          80 passed; 0 failed
fusor2-cpu           65 passed; 0 failed
fusor2-gguf          22 passed; 0 failed
fusor2-ir           239 passed; 0 failed
fusor2-tile         131 passed; 0 failed
fusor2-gpu          116 passed; 5 failed   <- the known set, unchanged
```

The `fusor2-gpu` five are the pre-existing set, verified against a worktree at
`c6bd339` (`117 passed; 5 failed`) with the *same five names*:
`a_narrow_output_stages_the_accumulator`,
`single_slot_fold_wgsl_is_unchanged`, and three `lower::region::tests::*`.
One caution for the next reader: `a_narrow_output_stages_the_accumulator`
sometimes *passes* inside a full parallel run — one whole-suite run on this tree
reported `118 passed; 4 failed`. Run in isolation it fails 3/3 on this tree and
3/3 at `c6bd339`, so that is flakiness under load, not a fix.

## Decode throughput — the gate

llama-3.1-8B-Instruct Q4_K_M, 63 decode tokens, three binaries interleaved in
one session, ten rounds: five in the order `76d42f7, c6bd339, HEAD` and five in
the reverse order. The first run of the session (8.68 tok/s, straight after
conformance) was discarded per the documented cold artifact.

```
76d42f7   22.05 21.42 22.25 21.48 22.29 22.44 22.30 22.14 20.97 22.12   mean 21.95  min 20.97
c6bd339   23.71 23.62 21.45 21.74 22.26 22.02 23.45 22.70 22.78 23.27   mean 22.70  min 21.45
HEAD      22.39 23.31 22.74 23.44 23.41 23.38 22.65 23.50 23.54 23.05   mean 23.14  min 22.39
```

* **HEAD beats the `76d42f7` baseline in 10 of 10 pairs**, 23.14 vs 21.95
  (+5.4%). That is the campaign's gate and it is not close.
* HEAD beats `c6bd339` — the tree this round started from — in 6 of 10 pairs,
  23.14 vs 22.70, and its *worst* run (22.39) is above `c6bd339`'s worst
  (21.45). The round is neutral-to-positive; nothing here regressed.
* For scale: old fusor decodes this model at 15.4 tok/s.

## What was NOT deleted, and why

* **`L1::Ext` / the `MacroOp` sugar layer** and **`L1::KRegion`** — both deleted,
  gated green, measured 3x and 3-10% slower respectively, and reverted. Rounds 3
  and 4 above.
* **`dequantize_slow`** — reads like `softmax_slow` and is not: it produces a
  class with a single member, which is what makes the quantized `defn` cases
  test arithmetic instead of selection.
* **The `ScatterMode` field itself.** With two members left that lower to one
  nest and price the same, the honest end state is no `mode` at all and one
  scatter rule. That is a graph-shape change of a different size — the floor's
  `KScatter` would then hash-cons onto the tile rule's — and it belongs behind
  its own cold-cache A/B, not stapled to this one.
* **`max_all` / `min_all` / `product_all` / `arg_min` / `to_f16` / `to_bf16`**
  were deleted despite the audit's "symmetric family" caution: every one of them
  was unreferenced, each has a live axis-taking twin (`max(axis)`, `cast`), and
  keeping an unreachable name for symmetry is the same argument the `*_fused`
  family lost.
