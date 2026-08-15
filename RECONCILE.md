# RECONCILE — what separates `fusor2` (mainline) from `fusor2-verified` (fork)

Scope: establish the divergence. **Nothing is ported by this document.** The one
file it moves is `DOCTRINE.md`, copied to `fusor2/DOCTRINE.md` as instructed.

Every claim below is backed by pasted output from this machine
(Apple M2 Max, Metal, 2026-08-15).

---

## 0. The topology nobody wrote down

`fusor2-verified/.git` is not a submodule checkout in the usual sense — it is a
**git worktree of a repo whose gitdir lives inside `fusor2/`**:

```
$ cat fusor2-verified/.git
gitdir: /Users/evanalmloff/Desktop/Github/training/fusor2/.git/worktrees/fusor2-verified

$ git --git-dir=training/fusor2/.git worktree list
/Users/evanalmloff/Desktop/Github/training/fusor2           6ddc007 [main]
/Users/evanalmloff/Desktop/Github/training/fusor2-verified  ef3f07d [decode-switch]
```

So `training/fusor2/` has **two** git identities:

* the outer `training` repo tracks its 217 files directly (branch `backprop`,
  content = commit `0a305bfa5`), and
* an inner repo at `training/fusor2/.git` whose `main` is `6ddc007` and whose
  second worktree is `fusor2-verified` on `decode-switch` @ `ef3f07d`.

**Consequence for the seal: deleting the `fusor2-verified` directory does not
delete the fork's history.** The 64 fork commits live in
`training/fusor2/.git`, which stays. But the reverse is also true — anything
that removes `training/fusor2/.git` (e.g. re-adding fusor2 as a fresh repo)
destroys the only copy of the fork's history. Prune the worktree registration,
never the object store.

### Divergence shape

`fusor2/` entered the outer repo as a directory import at `b22c042c9`; only
three outer commits have ever touched it:

```
$ git log --oneline --all -- fusor2
0a305bfa5 refactor sweep: -9.4k loc, dead plan cache and calibration deleted, ...
94f6a6a04 search-based extraction wip, session realize path, quantized lowering folded into rules
b22c042c9 wip search based
```

The fork branched from the inner `main` and is 64 commits ahead of it, 0 behind:

```
$ git --git-dir=training/fusor2/.git rev-list --count main..decode-switch
64
$ git --git-dir=training/fusor2/.git rev-list --count decode-switch..main
0
```

Sizes of the two independent bodies of work:

```
fork   (main..decode-switch, excluding vendor/):  150 files, +14,394 / -6,204   (net +8,190)
mainline (b22c042c9..0a305bfa5, fusor2/ only):    210 files, +11,411 / -21,228  (net -9,817)
files touched by BOTH:                            134
```

Total Rust in each tree today:

```
mainline: 113,338    fork: 129,695    (vendor/ excluded from both)
current on-disk difference: 219 differing/unique paths, ~54k unified diff lines
```

134 shared files is the number that decides the port strategy. Every hot file is
in it: `fusor2-cost/src/extract.rs`, `fusor2-cost/src/realize.rs`,
`fusor2-gpu/src/lower.rs`, `fusor2-gpu/src/target.rs`, `fusor2-gpu/src/launch.rs`,
`fusor2/src/session.rs`, `fusor2-tile/src/rules.rs`.

---

## 1. MAINLINE_HEALTH

### Conformance — 2 wrong values, and **the sweep caused them**

```
$ cd training/fusor2 && cargo run --release -p fusor2-conformance
816 results: 805 passed, 9 skipped, 2 failed
exit=1

FAILED  quantized::qmatmul_coop_shape_q4k [gpu]: item mismatch on gpu at [0, 1]: expected 108.16661, got 61.80017
FAILED  quantized::qmatmul_coop_shape_q5k [gpu]: item mismatch on gpu at [0, 1]: expected 150.69981, got 177.35925
```

The fork, same machine, same session, is green on the whole suite and on those
two rows specifically:

```
$ cd training/fusor2-verified && cargo run --release -p fusor2-conformance
784 results: 775 passed, 9 skipped, 0 failed
exit=0

ok      quantized::qmatmul_coop_shape_q4k [gpu]
ok      quantized::qmatmul_coop_shape_q5k [gpu]
```

(The fork reports fewer *results* because it deleted conformance rows that
re-tested one e-class against itself — `192a926`, `dbba115`, `76d42f7`. Fewer
rows, zero failures.)

**Two hypotheses tested, one disproved, one confirmed.**

*Disproved — it is not the missing vendored naga.* The fork's workspace pins
`[patch.crates-io] naga = { path = "vendor/naga" }` (the `CooperativeLoad`
bake fix); mainline's workspace has no `[patch]` section at all. Adding the
identical patch to mainline and rebuilding changes nothing — the failures are
bit-identical:

```
$ printf '\n[patch.crates-io]\nnaga = { path = "../fusor2-verified/vendor/naga" }\n' >> fusor2/Cargo.toml
$ cargo run --release -p fusor2-conformance qmatmul_coop_shape
FAILED  quantized::qmatmul_coop_shape_q4k [gpu]: item mismatch on gpu at [0, 1]: expected 108.16661, got 61.80017
FAILED  quantized::qmatmul_coop_shape_q5k [gpu]: item mismatch on gpu at [0, 1]: expected 150.69981, got 177.35925
12 results: 10 passed, 0 skipped, 2 failed
```

(reverted; `Cargo.toml`/`Cargo.lock` restored from backup.)

*Confirmed — mainline's own `-9.4k` sweep introduced them.* Checked out the
commit **before** the sweep into a scratch tree and ran the same filter:

```
$ git archive 94f6a6a04 fusor2 | tar -x -C /tmp/ml_presweep
$ cd /tmp/ml_presweep/fusor2 && cargo run --release -p fusor2-conformance qmatmul_coop_shape
ok      quantized::qmatmul_coop_shape_q4k [gpu]
ok      quantized::qmatmul_coop_shape_q5k [gpu]
12 results: 12 passed, 0 skipped, 0 failed
exit=0
```

and the **whole** pre-sweep suite is green, at the same result count mainline
reports today:

```
$ cd /tmp/ml_presweep/fusor2 && cargo run --release -p fusor2-conformance
816 results: 807 passed, 9 skipped, 0 failed
exit=0
```

Side by side — same tree, two commits apart, same machine, same naga:

| | results | passed | skipped | failed |
|---|---|---|---|---|
| `94f6a6a04` (pre-sweep) | 816 | 807 | 9 | **0** |
| `0a305bfa5` (mainline HEAD) | 816 | 805 | 9 | **2** |

Same machine, same unpatched naga, same cases: **green at `94f6a6a04`, wrong
values at `0a305bfa5`.** This is a DOCTRINE violation living in mainline HEAD.
The sweep's plausible blast radius on this path:

```
$ git diff --stat 94f6a6a04 0a305bfa5 -- <coop/quantized path>
 fusor2-gpu/src/emit/coop.rs       |  36 +-
 fusor2-gpu/src/emit/quantized.rs  |   2 -
 fusor2-gpu/src/lower/contract.rs  | 576 +++++++++++--------------------
 fusor2-tile/src/domains/coop.rs   | 178 ++++------
 fusor2-tile/src/rules/contract.rs | 125 +++----
 fusor2/src/composite/quantized.rs | 231 ++++---------
```

`lower/contract.rs` (−576-line rewrite) and `domains/coop.rs` are where to look.
Fix at source before anything is ported on top of it.

### Build

```
$ cd training/fusor2 && cargo build --workspace
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.06s
```

Clean.

### Can the four model crates build against mainline? **No — 145 errors.**

Repointed all four `models/*/Cargo.toml` at `../../fusor2/fusor2` in the working
tree (reverted with `git checkout -- models/`, nothing committed):

```
$ cargo check -p rbert
error[E0432]: unresolved import `fusor2::cache::MaskKind`
error[E0425]: cannot find function `stack` in crate `fusor2`
note: found an item that was configured out
  --> fusor2/fusor2/src/lib.rs:82:44
 81 | #[cfg(feature = "typed-api")]
 82 | pub use root::typed::{Device, Tensor, cat, stack};
error[E0107]: struct takes 0 generic arguments but 1 generic argument was supplied
  --> models/rbert/src/raw/qwen/model.rs:91:21
 91 |     pub(crate) cos: Tensor<2>,
error: could not compile `rbert` (lib) due to 145 previous errors
```

This is not incidental breakage — it is asset 2 missing. In mainline
`fusor2::Tensor` is still the **runtime-rank** tensor and the const-rank one is
behind `#[cfg(feature = "typed-api")]` (`fusor2/fusor2/src/lib.rs:79-82`); the
ported models are written against the fork's collapse where `fusor2::Tensor` **is**
the const-rank tensor (`fusor2-verified/fusor2/src/lib.rs:70-71`).

### Mainline decode tok/s: **not measurable**

There is no llama on mainline to measure. `fusor2/fusor2/examples/` contains
exactly one file, `vs_fusor1.rs`; there is no decode driver, and the model crate
that has one does not compile against mainline (above). Mainline also lacks the
substrate a per-token decode needs — `fusor2/src/cache/kv.rs` is 321 lines of
cat-only growable cache (`//! There is no capacity schedule or preallocated
backing store: an append` …) against the fork's 900-line version with the
fixed-capacity `Scatter{Set}` mode that makes one plan per token possible.

**Answer to "what is mainline's decode tok/s": there is no number, and getting
one is itself the port.**

The fork's number, for the bar step 8 has to clear (llama-3.1-8B-Q4_K_M,
`infer_local`, three consecutive runs immediately after the conformance suite):

```
$ for i in 1 2 3; do cargo run --release -p kalosm-llama --example infer_local \
    | grep -oE '[0-9]+\.[0-9]+ tokens/second'; done
=== run 1 ===
9.81 tokens/second      <- discarded: first run after a heavy GPU process
=== run 2 ===
19.70 tokens/second
=== run 3 ===
22.31 tokens/second
```

Run 1 landing at 9.81 is the documented environmental artifact, reproduced
exactly. This is a 3-run spot check to record the bar, **not** an A/B — no
paired interleave is meaningful when only one side can run.

---

## 2. ASSET TABLE

`ML` = `training/fusor2`, `FK` = `training/fusor2-verified`.

### Asset 1 — decode perf 6.3 → ~23 tok/s: **MISSING, all of it**

| commit | what | in mainline? | evidence |
|---|---|---|---|
| `243baef` byte traffic in `launch_work` | MISSING | ML `fusor2-cost/src/extract.rs:1019-1041` sums `macs + trans*8 + index_ops` and stops. FK `fusor2-cost/src/extract.rs:1697-1719` adds `BYTE_WEIGHT: u64 = 32` over `launch.bindings`. Mainline still has the exact macs-only gate the commit names: `extract.rs:456  if launch_work(graph, base, launch_ix) < min_macs` |
| `8c2ed31` lower-bound scan dedup | MISSING | ML `fusor2-cost/src/lower_bound.rs:109` has the plain "identical nodes share a scan" comment; FK `lower_bound.rs:368-434` adds the math-distinct-point dedupe ("the dedupe those scans exhausted `math_call_budget` on an 8B") |
| `6e708b4` fused quantized-row gather | MISSING | ML `fusor2-tile/src/rules/gather.rs` has 2 rules (`:14`, `:22`), both refusing quantized (`:75`, `:86`). FK `rules/gather.rs:25-30` registers a third, `apply = gather_quantized_rows`, defined at `:85` |
| `a772854` walk the L2 body as a DAG | MISSING | ML `fusor2-tile/src/liveness.rs` (843 lines) has no visit memo — `grep memo\|visited\|HashSet` returns nothing. FK `liveness.rs:462-480` carries `seen: FxHashSet<usize>` |
| `2912701`,`7394a1e`,`e068fe6`,`97a3706` artifact cache / build queue / per-launch keying / one-arm builds | MISSING | ML `grep -r artifact_cache` = **0 hits**; `fusor2-gpu/src/target.rs` is 592 lines against FK's 1,339. FK's whole explorer file `fusor2/src/session/explore.rs` does not exist in ML |
| `e7343a1` bind-group cache, verify memo, no per-launch DimBinding | MISSING | ML `fusor2-gpu/src/launch.rs:295 pub fn bind_group(...) -> Result<wgpu::BindGroup>` creates one per call. FK `launch.rs:260 bind_groups: Mutex<lru::LruCache<BindGroupKey, BindGroupEntry>>`, `:392` returns `Arc<wgpu::BindGroup>` |
| `c93634f` uniform pack once per resolve | MISSING | ML `fusor2-gpu/src/lower.rs:451 let pack = UniformPack::new(cx.plan);` per lower. FK `lower.rs:969 Arc::new(UniformPack::new(cx.plan))` shared, `:975` documents "the pack is a function of the *plan* alone" |
| `c6bd339`(+`b6f979b19` models) replay a fixed cache append | MISSING | ML `fusor2/src/cache/kv.rs` 321 lines, no `replay`. FK `cache/kv.rs` 900 lines, `:47 TensorCache::replay_append`, `:52` reused write-index leaves |

Also missing underneath all of it and not on the numbered list: the symbolic-length
decode substrate (`8811f4a`, `2f7d462`) and the whole sgemv kernel family
(`f63ea70`, `8129edc`, `076988d`, `7f21a1c`, `4d879b4`) — `grep -r
lower_sgemv_subgroup_cols` is **0 hits in mainline, 3 in the fork**. Asset 1 is
not eight patches; it is the back half of a 64-commit branch.

### Asset 2 — the typed public API: **MISSING**

`2f9974f`, `6cdc4aa`, `7a3cccd`, `76d42f7`, `4c5b32d`, `23155b4`, `3246232`, `aa6c070`.

```
ML fusor2/fusor2/src/lib.rs:40   typed-api = []            (Cargo.toml:40)
ML fusor2/fusor2/src/lib.rs:79   pub use root::dynamic::{Device, Tensor};
ML fusor2/fusor2/src/lib.rs:81-82 #[cfg(feature = "typed-api")] pub use root::typed::{Device, Tensor, cat, stack};

FK fusor2/src/lib.rs:70-71  pub use device::Device;
                            pub use tensor::typed::{Axis, Element, Minus1, Minus2, Tensor, cat, stack};
```

Mainline has no `API.md`, no `fusor2/src/api_surface.rs`, and still ships
`fusor2/src/ops/alias.rs` (71 lines) that `192a926` deleted. `aa6c070`'s seed
cycle repair is also absent: ML `fusor2-cost/src/extract.rs:124` does a bare
`argmin_member` per class; FK `extract.rs:1333` falls through to
`argmin_member_excluding` when two classes name each other.

### Asset 3 — the four model crates on the typed API: **DONE, and it is the constraint**

These edits are already committed in the outer repo (`82e688e07`, `a140e46b0`,
`9bbfc4baa`, `e1d0576d1`, `600d82c1e`, `a993d2b9c`). `Result<Tensor` threading in
`models/*/src`:

```
at 742bc1ce7 (pre-port): 90
today:                   13
```

The 13 survivors are all rank-parameterized (`Result<Tensor<2, F>>`,
`Result<Tensor<1>>`) on loader and fusor-v1 qwen-vision paths — the dyn
`Result<Tensor>` threading the port targeted is gone. **This asset is why
mainline cannot simply keep its own API:** the models are already committed
against the fork's surface, so either asset 2 lands in mainline or four model
crates get un-ported.

### Asset 4 — simplification: **MISSING (every item)**

| commit | mainline grep |
|---|---|
| `61a06c9` delete KMerged wave path | `KMerged` — **85 hits in ML**, 0 in FK. ML still has `fusor2-ir/src/rules/merge.rs` (350 lines) and `fusor2-gpu/src/lower/merged.rs` (1,137) |
| `e9126d0` delete `tile_contract` | `tile_contract` — 7 in ML (`fusor2-tile/src/rules/contract.rs:408`), 0 in FK |
| `4f0c3f4` delete `Family::GenericFold` | `GenericFold` — 9 in ML, 0 in FK |
| `192a926` delete the alias surface | ML `fusor2/src/ops/alias.rs` exists; `max_elementwise` 14 in ML vs 5 in FK (comments) |
| `dbba115` delete `softmax_slow` | `softmax_slow` — 16 in ML, 2 in FK |
| `fab718c` 36 unreferenced pub fns | not individually re-audited; the named survivors above are all present |
| `bf23ffe` never-selected scatter/gather modes | `WgPrivateMerge` 10 / `OneHotContract` 16 / `Vectorized` 10 in ML (`fusor2-tile/src/rules/gather.rs:84 gather_vectorized`), 3/4/3 in FK |

The one item the brief expected to be already-done is **half** already-done:
mainline's sweep deleted `fusor2-cost/src/calibrate.rs` (−1,032 in the sweep
diffstat) exactly as `f4a18ee` did, but mainline **kept**
`fusor2-cost/src/cache.rs` (83 lines of `$XDG_CACHE_HOME` helpers) which the fork
folded away. Treat `f4a18ee` as ~90% no-op.

### Asset 5 — documents: **MISSING**

`fusor2/` has `ARCHITECTURE.md` and `CONTRACTS.md` only. `DOCTRINE.md` (15
lines), `API.md` (277) and `SIMPLIFY.md` (1,141, including both committed
disproofs) exist only in the fork. `DOCTRINE.md` has been copied to
`fusor2/DOCTRINE.md` by this pass.

---

## 3. TRAPS_IN_MAINLINE

**Good news, stated precisely: mainline did not walk into any of the three
measured traps. The `-9.4k` sweep left every load-bearing node standing.**

**`L1::Ext` + the `MacroOp` sugar layer — PRESENT in mainline.** Both the node
and, more importantly, the union that makes it load-bearing:

```
ML fusor2/fusor2/src/composite.rs:388-419  pub(crate) fn macro_op(...)
        let sugar = g.add(Op::L1(L1::Ext { def: def.def_id(), ops: operands, attrs }))?;
        g.union(defn, sugar)
    })?;
    Ok(graph.tensor(root))
```

with the doc comment already stating the rule the fork measured the hard way —
"The returned `Tensor` is over the union root — returning either id alone would
pin one member of the class." `MacroOp::ALL` is intact at `composite.rs:131-140`,
all eight entries. So the `19.99 → 6.54 tok/s` cliff of `29c993e` is **not**
sitting in mainline.

One real difference, and it is a *porting hazard rather than a live bug*:
mainline unions with plain `g.union`, the fork with `union_stable`
(`FK composite.rs:456`, "a rebuild — a decode loop re-running the same model code
next step — gets the same id back instead of the class's *moved* root").
`grep -r union_stable` in mainline: **0 hits**. Plain `union` is correct for a
one-shot graph and wrong for per-token replay, so `031699c`/`3a9c171` must land
**before** any decode-replay work, not after.

**`L1::KRegion` — PRESENT in mainline.** `fusor2-gpu/src/lower.rs:1109`
(`L1::KRegion { .. } => merged::lower_kregion(...)`),
`fusor2-gpu/src/lower/merged.rs:366`, `fusor2-cpu/src/lower.rs:62`,
`fusor2-ir/src/saturate.rs:52 OpTag::KRegion => 15`. The 3-10% of `aca921d` is
not sitting in mainline either.

**`lower_sgemv_subgroup_cols` reading extents via `Ctx::dim_expr`
(20.9 → 1.75 tok/s) — not applicable yet.** Mainline has no sgemv-cols lowering
at all (`grep lower_sgemv_subgroup_cols` = 0). The trap becomes live the moment
`f63ea70` is ported; carry the baked KV length with it.

**`sink::fold_operand_views`'s `reads_its_view_densely` guard — PRESENT and
intact in mainline** (`fusor2-ir/src/rules/sink.rs:199` call, `:251`
definition). Mainline is *stricter* than the fork on the neighbouring condition
— it refuses every multi-node spine (`sink.rs:203  if spine.views.len() != 1`)
where the fork composes them (`FK sink.rs:232-237`, the multi-view spine folding
of `7669767`). That is a fork asset to port, not a mainline bug, and porting it
must not touch the density guard.

**What *is* in mainline that the traps section should have warned about: a
wrong-value miscompile mainline's own sweep introduced** (section 1). By the
general rule this fork established — "never selected does not mean safe to
delete; safe means no call sites at all" — a sweep that deleted 21,228 lines
across 203 files and turned two green quantized coop rows red is the same class
of mistake the fork's disproofs were written to prevent, caught by the suite
instead of by a decode measurement. **This is the highest-priority item in the
reconciliation.** It is also a blocker: any A/B measurement taken on a mainline
that miscomputes q4k is measuring the wrong program.

---

## 4. PORT_PLAN

### The finding that decides the strategy

The brief assumes "port five assets into mainline". The trees do not support
that shape. The fork is 64 commits and +8,190 net non-vendor lines ahead of the
shared base; mainline is 210 files and −9,817 lines ahead of it on a *different*
axis; **134 files were rewritten by both sides**, including every file asset 1
touches. `git cherry-pick` across the two object stores is available (they share
`training/fusor2/.git`) but every hunk in `extract.rs`, `realize.rs`, `lower.rs`,
`target.rs`, `launch.rs` and `session.rs` lands on context the sweep rewrote.

Ledger of what each side uniquely owns:

| | mainline-only | fork-only |
|---|---|---|
| commits | 3 (import + wip search-based + sweep) | 64 |
| net LOC | −9,817 | +8,190 |
| conformance | 805 pass / **2 wrong values** (807/0 before the sweep) | 775 pass / **0 failures** |
| models build | no (145 errors) | yes (committed, in use) |
| decode | no driver, no substrate | 19.70 / 22.31 tok/s measured today |
| unique artifacts | `fusor2-tile/src/lower.rs` (shared cpu/gpu tile builder, 235), scalar walker combinators in `fusor2-ir/src/scalar.rs`, egraph memo + saturation clone removal, search-based extraction wip, and the deletion of `calibrate.rs` | everything in section 2 |

**Recommendation: invert the direction.** Land mainline's three unique wins
*onto the fork*, then move the fork's content into `training/fusor2/` and drop
the `fusor2-verified` path. The user's stated end state is preserved exactly —
one tree at `training/fusor2`, submodule gone — but the merge is ~3 commits of
refactor replayed onto a green tree instead of 64 commits of measured perf work
replayed onto a tree that is red, has no decode driver, and cannot build its own
models. If the direction must stay as briefed, the plan below is the same list
walked backwards, and steps 0-2 are identical either way.

### Dependency order

**Step 0 — fix mainline's miscompile at source. Blocking, nothing measured
before it.** Bisect `94f6a6a04..0a305bfa5` inside `fusor2-gpu/src/lower/contract.rs`
and `fusor2-tile/src/domains/coop.rs` for `qmatmul_coop_shape_q4k`. Never gate,
never shape-special-case. Add nothing to conformance — the row already exists and
already catches it.

**Step 1 — documents and the naga pin, no code risk.** `DOCTRINE.md` (done),
`API.md`, `SIMPLIFY.md`; move `vendor/naga` under `fusor2/` and add
`[patch.crates-io]` to `fusor2/Cargo.toml`. The fork's own workspace has this and
mainline does not, so mainline's coop f32 path is running unbaked
`CooperativeLoad` today. It is *not* the cause of the two failures (proved in
section 1) but it is the coop NaN flake, and it is one line.

**Step 2 — e-graph identity primitives.** `031699c` + `3a9c171`: `union_stable`,
stable member ids, pointer-free 128-bit body hash. Everything downstream assumes
a rebuild returns the same ids. Cheap, self-contained, touches
`fusor2-ir/src/egraph.rs` + `fusor2/src/composite.rs`.
*Conflict:* low — mainline's sweep touched `egraph.rs` (memo/clone removal) but
not the union path.

**Step 3 — the typed API (asset 2).** `2f9974f`, `6cdc4aa`, `76d42f7`,
`4c5b32d`, `23155b4`, `3246232`, `aa6c070`, `7a3cccd`. Do this *before* asset 1:
it is the only step whose success is objectively checkable without a GPU
measurement (`cargo check -p rbert -p rwhisper -p kalosm-llama -p
segment-anything-rs` against `../../fusor2/fusor2` must reach 0 errors), and it
unlocks the decode driver every later step needs to measure with.
*Conflict:* medium-high. Mainline's sweep rewrote `fusor2/src/tensor/typed.rs`
(−234) and `lib.rs` (−85) independently. Expect to re-derive rather than apply.

**Step 4 — asset 4 deletions.** `61a06c9`, `e9126d0`, `4f0c3f4`, `192a926`,
`dbba115`, `fab718c`, `bf23ffe`. Deliberately before the perf work: they remove
~5k lines of the same files asset 1 edits, so doing them first shrinks the
conflict surface of every later step. Re-run each deletion's census on
*mainline's* graph before deleting — mainline's search-based extraction may
select a member the fork's never did, and the rule that governs here is the
fork's own: **safe means no call sites at all, not "never selected".**
*Conflict:* `61a06c9` collides head-on with mainline-only
`fusor2-ir/src/rules/merge.rs` (350) and `fusor2-gpu/src/lower/merged.rs` (1,137),
which the sweep just rewrote; `KRegion` must survive that deletion intact
(`aca921d`).

**Step 5 — the decode substrate.** `8811f4a`, `2f7d462`, `c6bd339` (fixed-capacity
symbolic-length KV cache, one plan per token, replayed append) plus the outer
repo's `83105b744`/`b6f979b19`. Largest single lift: `fusor2/src/cache/kv.rs` goes
321 → 900 lines and `session.rs` 1,476 → 2,111.
*Conflict:* high — `session.rs` was 60% rewritten by the sweep (−347).

**Step 6 — host-side decode perf.** In the fork's own order, which is also
dependency order: `a772854` (DAG walk) → `2912701` → `7394a1e` → `e068fe6` →
`97a3706` → `e7343a1` → `c93634f`. These all key off the artifact cache
`2912701` introduces; none can be taken alone.
*Conflict:* high — `fusor2-gpu/src/target.rs` more than doubles.

**Step 7 — kernel and cost-model perf.** `f63ea70`+`8129edc`+`076988d`+`7f21a1c`+
`4d879b4` (sgemv family — **carry the baked KV extent, do not re-derive via
`Ctx::dim_expr`**), `243baef`, `8c2ed31`, `6e708b4`, `7669767` (multi-view spine
folding — do not relax `reads_its_view_densely` while doing it).

**Step 8 — the seal.** Only now: repoint all four `models/*/Cargo.toml` at
`../../fusor2/fusor2`, `cargo check` all four to 0 errors, run the full
conformance suite to 0 failures, and A/B decode against the fork binary —
paired interleaved wall-clock, ≥5 pairs, reversed order in a second set, first
run after any heavy GPU process discarded, extracted with
`grep -oE '[0-9]+\.[0-9]+ tokens/second'`. Parity with `ef3f07d`, not "close".
Then, and only then, `git worktree remove` the fork path — leaving
`training/fusor2/.git`'s objects alone.

### Drop as already-done

* `f4a18ee`'s `calibrate.rs` half — mainline deleted the same 1,032 lines
  independently. Its `cache.rs` half is still open (83 lines).
* Nothing else on the list is already-done. Every other numbered item was
  verified present-in-fork / absent-in-mainline above.
