# SIMPLIFY — second-pass deletion census

Census phase only. **Nothing was deleted for this document.** Every claim below
is backed by pasted output from a token scan over both workspaces, a compile
experiment (run and then reverted), or a call-graph fact. The first campaign's
ranked list is exhausted (its seal is commit `ef3f07d`; −3,440 net Rust lines,
decode 21.95 → 23.14); this is a fresh census over the six angles the second
pass named.

Committed tree at the time of measurement: `decode-switch`, HEAD `ef3f07d`.

> **Caveat — a concurrent session was editing this checkout during the
> census.** While these measurements ran, another session was actively
> trimming doc comments across the tree (working-tree diff grew from clean to
> `182 files changed, 415 insertions(+), 1957 deletions(-)` over the course of
> the census, all mtimes current). All scans below were confirmed against
> content that is identical in HEAD unless noted. Two side-effects of that
> in-flight work showed up as fresh rustc warnings and are **not** census
> findings — recheck them after that session lands:
> `fusor2-gpu/src/target.rs:197` `static LAST_EXIT is never used` and
> `fusor2-gpu/src/target.rs:408` `unused variable: gap`.

---

## How the census was taken

1. **No-call-site sweep, rerun.** Token-frequency scan (comments stripped)
   over every `.rs` in this repo plus `training/models` and
   `training/interfaces`: 2,163 `pub fn` definitions, plus all pub
   struct/enum/trait/type definitions. A name whose count is 1 is referenced
   nowhere; count 2 was classified by hand (the one extra mention is a real
   call site, a rule registration, a string literal, or a test of the item
   itself).
2. **rustc dead-code**: `cargo check --workspace --all-targets --all-features`
   after touching every crate root — **zero warnings** on HEAD content. The
   five `allow(dead_code)` sites were audited by hand (four are justified
   test-only/type-level items; one is a real dead fn, item 7).
3. **Deps and features**: no cargo-machete/udeps installed; manual scan of
   every `[dependencies]` entry against `ident::`/`use ident` in that crate's
   sources, then **proof by construction**: all suspects removed at once,
   `cargo check --workspace --all-targets` green, reverted.
4. **Cross-crate duplication**: fn-name intersection across the five compiler
   crates, then body diffs on the plausible pairs.
5. **Conformance self-duplication** and **the Dyn surface**: covered by the
   same token scan (a Dyn method with no typed wrapper and no caller would
   have count 1 — there are none) plus a grep for `_slow`/`_alias`/`agrees`
   case families.

## Ranked plan

Evidence classes: **NCS** = no call site (grep + classification; compile
experiment where noted), **DEP** = dead dependency proven by removal + build,
**FEAT** = feature gating nothing, **DUP** = duplicate proven by diff,
**A/B** = member-minting or perf-relevant, gated on a paired interleaved A/B.

| # | candidate | class | est. LOC | risk |
|---|---|---|---|---|
| 1 | `graphvis_dot` + its subject test (`fusor2-gpu/src/launch.rs:918`) | NCS (self-test only) | ~60 | low; **debug tooling** — confirm the DOT dumps of commit `4e3261a` don't route here (grep says they don't) |
| 2 | `check_output` + subject test (`fusor2-conformance/src/goldens.rs:107`) | NCS (self-test only) | ~35 | low; output-hash goldens are otherwise unwired — decide wire-or-delete |
| 3 | `cases_from_rows` + subject test (`fusor2-conformance/src/harness.rs:202`) | NCS (self-test only) | ~30 | very low |
| 4 | `d_expr` + subject test (`fusor2-autograd/src/map_adjoint.rs:472`) | NCS (self-test only) | ~25 | low; reads like a custom-adjoint API entry — check intent |
| 5 | `run_case` (`fusor2-conformance/src/harness.rs:482`) | NCS (only other mention is a string in `suite.rs:542`) | ~22 | very low; fix the error-message string |
| 6 | `fill_indices` + subject test (`fusor2-conformance/src/harness.rs:341`) | NCS (self-test only) | ~20 | very low |
| 7 | `layout_elements` (`fusor2-gpu/src/lower.rs:77`, already `#[allow(dead_code)]`) | NCS | ~15 | very low |
| 8 | `reverse_order` + subject test (`fusor2-autograd/src/backward.rs:207`) | NCS (self-test only) | ~15 | very low |
| 9 | `TensorCache::fixed_windowed` (`fusor2/src/cache/kv.rs:118`) | NCS (self-test only) | ~10 | low; the ring test rebuilds via the `fixed_named` + `window` path `KvCache::windowed` already uses. The ring feature itself is **live** (`models/kalosm-llama/src/raw/cache.rs:45`) |
| 10 | `Reverse::topology` accessor (`fusor2-autograd/src/backward.rs:145`) | NCS (only other mention is inside an error string) | ~5 | very low |
| 11 | 10 unused dependency entries (list below) | DEP | 10 TOML | none — proven by removal + build |
| 12 | `fusor2`'s `cpu`/`gpu` features (`fusor2/Cargo.toml:26-29`) | FEAT | 4 TOML | none — zero `cfg(feature = "cpu"/"gpu")` anywhere in either workspace; removal + check green in both (pasted below) |
| 13 | `distribute_workgroups` twins (`fusor2-cost/src/realize.rs:699` vs `fusor2-gpu/src/lower.rs:257`) | DUP → A/B | ~20 net | medium: bodies differ in width (u64/u32) and z-clamping; consolidation must keep the **gpu** semantics bit-exact and be proven by `FUSOR2_DUMP_PLAN` byte-identity, not by sampling |
| 14 | pinned-but-caller-free typed methods: `zeros_dims`, `resize_dims`, `broadcast_dims`, `rope_at` | NCS but **on the API.md surface** | ~55 | surface decision, not a dead-code deletion — API.md names all four (`API.md:167,179,205`). Precedent from round 5: pinned = keep unless the surface itself is renegotiated |

**Totals: NCS ~237 LOC (items 1-10) + ~55 pinned (item 14, decision needed);
DEP 10 TOML lines; FEAT 4 TOML lines; DUP ~20 LOC (A/B-gated).**

The honest headline: after two campaigns (−4.8k, then −3.4k net lines) the
no-call-site class is nearly mined out. The count==1 sweep that found 43
orphans last time now finds **zero** — every candidate above came from the
count==2 classification (self-test-only and string-literal-only references),
which is the next stratum down, and it is thin.

### Item 11 — the ten dead dependency entries

```
fusor2-conformance/Cargo.toml:16  fusor2-autograd
fusor2-conformance/Cargo.toml:19  fusor2-gpu
fusor2-conformance/Cargo.toml:20  fusor2-cpu
fusor2-conformance/Cargo.toml:21  smallvec
fusor2-conformance/Cargo.toml:25  rand
fusor2-cpu/Cargo.toml:14          fusor2-cost
fusor2-gguf/Cargo.toml:16         bytemuck
fusor2-gpu/Cargo.toml:19          bytemuck
fusor2-tile/Cargo.toml:15         fixedbitset
fusor2-tile/Cargo.toml:16         half
```

Grep for each crate ident in the owning crate's sources returns nothing (the
`rand`/`half` textual hits are the English words "operand"/"half" in doc
comments). Proof by construction — all ten removed at once:

```
$ cargo check --workspace --all-targets
    Checking fusor2 v0.1.0 (.../fusor2)
    Checking fusor2-conformance v0.1.0 (.../fusor2-conformance)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 8.20s
```

Reverted after the experiment. Note `fusor2-conformance` still reaches both
backends through `fusor2` itself; dropping its direct deps changes no feature
resolution (the workspace has no non-default features in play).

### Item 12 — the `cpu`/`gpu` features

Declared and defaulted in `fusor2/Cargo.toml`, they gate nothing: zero
`cfg(feature = "cpu")`/`cfg(feature = "gpu")` in either workspace, no optional
deps tied to them, and no downstream `features = [...]` enables them. Removal
experiment:

```
$ cargo check --workspace --all-targets        (fusor2 repo)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.79s
$ cargo check -p kalosm-llama                  (training repo)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 9.95s
```

Reverted. `fork-metal` is **not** on this list: it is a live experiment switch
(enabled from the command line) whose two capability arms are documented in
`fusor2-gpu/src/caps.rs`, and the vendored naga fork it pairs with is
off-limits by standing order.

### Item 13 — `distribute_workgroups`

Same slab-count-first algorithm, two homes. Differences that make this a
careful consolidation rather than a copy-delete: the cost copy works in `u64`
with `z` clamped to `max`, the gpu copy in `u32` with a `total <= max` early
return and unclamped `z`. `fusor2-gpu` already depends on `fusor2-cost`, so
the single home is `fusor2-cost` carrying the gpu semantics; the cost model
then prices exactly the grid the backend launches, which is a small coherence
win on top of the deletion. Gate: `FUSOR2_DUMP_PLAN` byte-identity over a full
decode, plus the standard round gates. `scalar_element` looked like the same
pattern and is **not** — see rejected leads.

## Leads checked and rejected this pass

* **`scalar_element` (cost vs gpu) is a deliberate divergence, not a
  duplicate**: cost maps `Dtype::Q(_)` → `ScalarElement::F32` (what a
  dequantized value stages as), gpu maps it → `U32` (what the packed words
  bind as). Consolidating would miscompile one side. Leave both; a comment
  cross-linking them would not hurt.
* **`grid_for` (gpu vs cpu)**: same name, different jobs (gpu records a
  `GridSpec` for artifact replay; cpu is a bare div_ceil). Not duplicates.
* **Test fixtures duplicated across crates** (`plan_for`, `coop_case`,
  `gpu_caps` ×2 each): cross-crate test-util consolidation would add a crate
  to delete ~40 test lines. Not worth the surface.
* **`point_scatter`, `graph_mut`**: count==2, but the one extra reference is a
  *utility* use inside a real test of something else, not a test of the item.
  Live.
* **Conformance self-duplication (angle 5)**: the alias-delegation class died
  with the alias layer in round 5. What remains at the `agrees` level —
  `welford_agrees_with_the_two_pass_variance`,
  `fold_split_agrees_when_reassoc` — tests real numeric properties, and
  `dequantize_slow` stays per the round-5 record (its class has no
  `Logical::Dequant` alternative, so the case tests arithmetic, not selection).
* **The Dyn surface (angle 4)**: no Dyn method has count 1. Everything the
  typed facade doesn't wrap is called directly by conformance, the examples,
  or the models. Nothing to delete.
* **rustc dead-code (angle 1, non-pub)**: zero warnings across
  `--all-targets --all-features` on HEAD content. The five `allow(dead_code)`
  sites: `api_surface.rs` (the pin file, by design), two type-level assertions
  in `typed.rs` ("the assertion is that it type-checks"), `lane_offsets`
  (`cfg_attr(not(test))`, read by the coverage test), and `layout_elements` —
  only the last is real dead code (item 7).
* **`fusor2-cpu` vs `fusor2-gpu`/`fusor2-tile` structure (angle 6)**: the
  fn-name intersection surfaced no body-level duplicates beyond items 13 and
  the rejected pairs above; `fusor2-cpu/src/lower/contract.rs` (1,033 lines)
  parallels the gpu's (2,283) in shape but lowers to a different execution
  model (tape/registers, not naga). Serving both from `fusor2-tile` is a
  rearchitecture, not a deletion; nothing census-sized here. It stays
  load-bearing for dual-backend conformance either way.
* **The e-graph member surface**: untouched by design. Both historical
  disproofs below stand; nothing that mints or unions members is on this
  round's list at all (item 13 is a scalar helper, not a node).

## Baseline (for this round's seal to quote against)

`tokei . --exclude vendor --exclude target`, committed HEAD `ef3f07d`:

```
 Rust                  207       113048        98464         6123         8461
 Total                 222       115419        98697         7778         8944
```

(113,048 total Rust lines, 98,464 code — down from 116,059/101,165 at the
first census.)

Decode, llama-3.1-8B-Instruct Q4_K_M, 63 tokens, binary built at `ef3f07d`,
six consecutive runs in one session:

```
run0: 9.06 tokens/second     <- discarded: first run after a heavy GPU process
run1: 22.29 tokens/second
run2: 19.00 tokens/second
run3: 18.97 tokens/second
run4: 19.19 tokens/second
run5: 20.98 tokens/second
```

Baseline: **median 19.19, mean 20.09, range 18.97-22.29** over the five kept
runs. Same discipline as before: a change must win a ≥5-pair interleaved A/B
(both orders when ambiguous), or prove plan byte-identity via
`FUSOR2_DUMP_PLAN`.

---

# Historical disproofs — do not retry, kept from the first campaign

These two are the reason the governing rule exists: **in this e-graph
compiler, "never selected" does not mean safe to delete.** Union members are
the extractor's choices; both of these were lowered zero times on every
workload and both cost real throughput when removed.

### `Launch::Ext` and the `MacroOp` sugar-node layer — **REJECTED, do not retry**

> **The audit was right that `Ext` is never selected and wrong that deleting it
> is free.** The whole layer was deleted (−2,059 code lines), every gate passed,
> and the shipped decode went **19.99 → 6.54 tok/s**. Reverted.

**Deadness re-proved, three independent ways.** All eight `MACRO_OPS` declare
`lower_per_target: &[]`; `verify_plan::check_extensions` errors on any selected
node with an empty row, and `LocalSearch` calls it on **every** plan it
returns — so a single `Ext` selection would fail the run outright. Neither
backend ever installs a registry outside its own tests. A direct probe over the
full suite agreed:

```
806 results: 797 passed, 9 skipped, 0 failed
PROBE_EXT_SELECTED: 0   PROBE_EXT_LOWER_GPU: 0   PROBE_EXT_LOWER_CPU: 0
```

**Why it cannot go.** `Ext` is dead as a *selectable node* and load-bearing as
a *union partner*. `macro_op` ends with `union_stable(defn, sugar)` and hands
the caller the **`Union` spine id**; with the sugar gone there is no second
member, no union, and every composite hands back a bare `Logical` member id
instead. Fusion quality collapses on that difference. Isolated to that one
line at `61a06c9`, with `Launch::Ext` still minted and everything else untouched:

```
-    let root = graph.union_stable(defn, sugar)?;
-    Ok(graph.tensor(root))
+    Ok(graph.tensor(defn))
```

| llama-3.1-8B-Q4_K_M, 63 tokens | tok/s | Launch::Map | Launch::Fold | Launch::Contract | Sgemv | Sgemm | Coop |
|---|---|---|---|---|---|---|---|
| `61a06c9` unmodified | **19.99** | 4798 | 250 | 1144 | **572** | 0 | 0 |
| one line above, nothing else | **6.46 / 6.57** | 6750 | 982 | 412 | 102 | 102 | 106 |
| full `Ext` deletion | **5.75 / 6.32 / 6.54** | 6750 | 982 | 412 | 102 | 102 | 106 |

The last two rows are byte-identical histograms, which is the isolation:
**none of the ~2,000 deleted lines cost anything; returning a member id
instead of a spine id costs 3x.** The decode matmuls stop lowering as `Sgemv`
— the right kernel at `M = 1` — and fall back to a generic `Launch::Fold`/`Launch::Map`
reduce plus some `Sgemm`/`Coop`, which at `M = 1` is pure waste. It is not
compile-side: both binaries report `saturate (skipped) … replay hit` on every
token, and the deletion *lowers* the node count (46,978 → 46,720).

**What this actually is.** A compiler fragility, not a fact about `Ext`: which
kernels get fused depends on whether the caller happens to hold a `Union` id
or a member id of the same class. Rules match spines (`Views(vs, X)`,
`trace_pure_views`), and a `Union` in the operand chain is what lets them.
Deleting `Ext` is blocked behind fixing *that* — make fusion match on class
membership rather than on the caller's chosen id — which is a rewrite-layer
change, not a deletion.

### `Launch::Launch::Region` — **DELETED, MEASURED, REVERTED — do not retry**

> The node is exactly as dead as the census said — proposed thousands of
> times, lowered zero times — and deleting it is still a 3-10% decode
> regression. Patch kept at `/tmp/kregion.patch` (19 files, 76 insertions,
> 1,517 deletions; 1,441 net Rust lines).

**The deadness claim is CONFIRMED**, re-proved with a fresh probe — a counter
at `form_kregion` past its operand search and at both backends' `Launch::Region`
lowering arm:

```
conformance (806 results, CPU+GPU): PROBE_KREGION_MINT 940
                                    PROBE_KREGION_LOWER_GPU 0
                                    PROBE_KREGION_LOWER_CPU 0
llama decode + rbert (one log):     PROBE_KREGION_MINT 450
                                    PROBE_KREGION_LOWER_GPU 0
                                    PROBE_KREGION_LOWER_CPU 0
```

**The deletion gated completely green** — both workspaces built, conformance
identical to HEAD, `FUSOR2_VERIFY_MEMBERS=1` quantized 108/108, every crate's
unit tests at their known state. **And it is 3-10% slower.** Interleaved
decode, both orders, 15 pairs:

```
HEAD  22.34 21.27 22.06 22.02 22.00 21.93 21.95 21.87 22.09 22.10 21.82 21.84 22.11 21.66 21.48
MINE  20.44 21.53 21.13 21.34 21.38 21.79 20.53 21.98 21.96 21.65 20.81 22.31 21.41 20.57 19.67
```

mean **21.90 vs 21.23** (−3.1%), MINE loses **12 of 15 pairs**. Re-run with a
private, equally cold `XDG_CACHE_HOME` per binary — the only uncontaminated
form of this measurement:

```
cold MINE  18.87  18.96  19.88      mean 19.24
cold HEAD  21.26  21.11  21.61      mean 21.33     -9.8%, 3 of 3 pairs lost
```

The two private caches came out **identical**, so the tuner is not the
difference. Nor are the kernels: `FUSOR2_WGSL_DUMP` produced **506 shaders
each, identical filenames, identical concatenated md5** — every kernel either
binary can dispatch is byte-for-byte the same. Something above the kernel
universe (which member each launch adopts, or per-token host work) got worse.

**The lesson, which generalizes.** A node can be lowered zero times and still
be load-bearing as a **waypoint the extractor passes through** —
`FORM_KREGION` unions a `Launch::Region { members: [producer, fused] }` into the
fold's own class, so it is an alternative every `RESELECT`/`FLIP` move can see
even though no plan ships it. **"Proposed but never selected" is not evidence
that a node is free to delete.** Anything that mints e-graph members is gated
on a paired cold-cache A/B *before* its code is touched, not after.
