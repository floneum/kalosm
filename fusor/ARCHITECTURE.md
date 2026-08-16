> Working target: `fusor/`, this directory.
> Acceptance target: betlang's `trainer/`, which is **not in this repository**. The gate below was enforced from `fusor-conformance` while fusor lived beside that crate; the MSQ1 and trainer-gate cases did not survive the move here (see `fusor-conformance/src/lib.rs`). The numbers stay as the design's stated bar; nothing in this checkout checks them.

# fusor — Architecture

## 0. Thesis and acceptance

Every phase-ordering failure catalogued in fusor is one bug: **a decision written into a data structure that the next decision cannot un-write.** `MatMulParams` on the operation. `commit_recognized` deleting the composed form. `LayoutPass`'s `unwrap_or_else(Layout::contiguous)`. `merge_profile` returning `None` for any matmul carrying an epilogue. `tile_probe_group`, chosen because allocation runs before the merge partition.

fusor removes the possibility rather than managing it. There is one acyclic e-graph into which the frontend, autograd and every lowering rule only ever *add*. Kernel family, tile geometry, split-K, staging depth, epilogue fusion, horizontal merging, layout, materialization, rematerialization, allocation and shape specialization are all selections in **one extraction**, priced by **one scalar picosecond cost function evaluated on the realized DAG**.

Three levels: **Logical `tensor`**, **Launch `nest`**, **Kernel `tile`**. The e-graph spans Logical and Launch. Kernel is the per-kernel IR produced *after* extraction; it carries its own verifier and exactly two closed-form argmins (§1.3).

The acceptance test is not elegance. It is that `trainer/` runs **unmodified** at >= 11,417 examples/sec epoch-average, teacher parity >= 0.9483, and a **byte-identical 47,840-byte MSQ1 export**, with all four of its documented workarounds deleted: no shape padding, no `[1]`-tensor scalars, no hand-written embedding backward, no reshape-as-maxpool.

---

## 1. The levels

### 1.1 Logical `tensor` — ten nodes

Whole-tensor algebra. No index space, no loop, no device. Where the user's program arrives and where autograd runs.

```rust
pub enum Logical {
    Leaf(LeafKind),   // Buffer | Param | Const(Splat) | Uniform(SymId) | Quantized(QFmt, QLayout)
    Map      { expr: ScalarExpr, ins: SmallVec<[Id; 4]>, outs: u8 },
    Fold     { carrier: Carrier, axis: u32, acc: Dtype, ins: SmallVec<[Id; 4]> },
    Contract { spec: EinSpec, acc: Dtype, a: Id, b: Id },
    Restride { specs: SmallVec<[StrideSpec; 6]>, x: Id },
    Window   { specs: SmallVec<[SlidingWindow; 3]>, x: Id },
    Gather   { axis: u32, x: Id, idx: Id },
    Scatter  { axis: u32, combine: ScatterCombine, base: Id, idx: Id, upd: Id },
    Dequant  { fmt: QFmt, layout: QLayout, x: Id },
    Project  { slot: u8, x: Id },
}
```

`ScalarExpr` is a hash-consed tree over a closed vocabulary: 21 unary math functions, 15 binary, 6 compares, `select`, `cast`, `round/floor/ceil/trunc(RoundMode)`, `bitcast`, `dot`, `splat`, `Arg(u32)`, `Lit`, `Uniform(SymId)`, `IndexOf(axis)`. **All 23 elementwise unaries, 8 scalar-arith unaries, 12 comparisons, `where_cond`, `clamp`, `relu`, `sigmoid`, `silu`, `gelu` and `tanh_exact` are one `Map` with a different expression.** `NaryOp`'s 50-variant discriminant — whose ordering the reference admits is load-bearing in kernel cache keys — does not exist.

`Project` plus `outs: u8` on `Map`/`Contract` gives tuple results, which `attention_grads`' combined `[B,H,2*Lk,Dh]` dk/dv buffer and `rope_pair_fused` require.

**`Carrier` is the fold algebra, as data** — `slots: [SlotTy]` (`Scalar | Vector(Dim)`), one `identity: Splat` per slot, a `lift` from the fold's operands into the accumulator and an associative `merge` of two accumulators, both ordinary `ScalarExpr`s. `Add`, `Mul`, `Max` and `Min` are `Carrier::binop` **values**; there is no named-combine enum, so `(n, mean, m2)`, `(max, sum)` and attention's `sum p*v` are things a *rewrite* constructs rather than variants the core enumerates. Naming an algorithm in the core is what this deletes: the previous `Combine::OnlineSoftmax` put the name in the type system while the GPU lowering resolved one `TileReduceOp` for the whole fold and updated only `accs[0]`, so `Fold{OnlineSoftmax}` computed `max(x)` and discarded the sum. A carrier's lanes are appended to the output shape, so slot readback is an ordinary `Restride` and no new node kind appears. `TiePolicy` survives as `Carrier::tie`, read only by `fold_adjoint`: an autograd attribute, never a compiler decision.

**Types.**

```rust
pub struct ValueFacts {
    pub dtype: Dtype,                  // F32 | F16 | BF16 | U32 | I32 | Q(QFmt)
    pub shape: SmallVec<[Dim; 6]>,     // Dim = Const(u64) | Sym(SymId)
    pub numeric: NumericContract,      // { min_accum_bits: u8, reassoc: bool, contract: bool }
    pub persistence: Persistence,      // Step | Persistent
}
```

Rank is runtime data. `Dim::Sym` is a symbolic extent bound at dispatch, never at compile — this is the whole of trainer constraint 1.

Two widenings past the reference's `{f32, f16, u32}`, both paid for: **I32** (float→int casts, `round`, sort-key scatter lowering) costs one `ScalarElement` row; **BF16** is free, sharing the `widen-compute` lowering rule with F16. Booleans stay 1.0/0.0 in the operand dtype, preserving `where_cond` and the comparison surface verbatim.

`NumericContract` is **monotone**: no rewrite may lower `min_accum_bits` or enable `reassoc`/`contract` where a value forbids it. This single fact does three jobs. It makes `fold_split` sound (float `Add` is not associative; split-K, online softmax and Welford are *not* value-equal to their unsplit forms on an f16 accumulator). It kills the trainer's `round_small` — 14 chained comparisons existing only because Metal's default fast math may fold the `(x + 2^23) - 2^23` trick — by giving `round(x, HalfAwayFromZero)` a `reassoc: false` value that survives to WGSL as an emitter obligation. And it keeps the coop path's f32 accumulator from being narrowed by an epilogue fusion.

`Persistence` distinguishes step-local activations from model weights. It is what makes quantized repack amortize against a weight's lifetime rather than a global constant, and it tells the extractor which buffers it may legally recompute.

**verify_l0.**
1. Shape/dtype inference is total; every op supplies `fn infer(&[ValueFacts]) -> Result<ValueFacts>`.
2. **No implicit broadcasting.** All `Map` operands have identical shape; the frontend emits `Restride { multiplier: 0 }`. Right-aligned broadcast rules (source dim consumed when equal or 1; unmatched target dims insertable at any position; unconsumed source dim is an error) live in the frontend.
3. `Fold`: `axis < rank`; the carrier's slot vectors agree in length; every identity is a value of `acc`; every `Vector` slot extent is constant; and **`merge(identity, identity) == identity`** — the obligation that catches a rescale spelled without a delta guard, which computes `0 * exp((-inf) - (-inf)) = NaN` on every schedule that merges padded identity lanes.
4. `Contract`: every `EinSpec` label appears in >= 2 of {a, b, out}; contracted extents agree; `acc.bits >= numeric.min_accum_bits`.
5. `Restride` composes `StrideSpec { input_dim, multiplier, size, offset }` **relative to current strides** (verbatim from the reference). Where `Const` dims make bounds decidable it is checked statically; where `Sym` dims make it undecidable a **runtime mask obligation is recorded on the node** and discharged by codegen. There is no third case, and there is no user `assume`: this is what preserves the invariant licensing `create_shader_module_trusted`.
6. `Scatter{Set}` with possibly-duplicate indices is rejected unless the node carries `unique: true`. `Scatter{Add}` is always legal and duplicates accumulate (normative — the embedding table receiving one token twice gets the summed gradient).
7. `Dequant`: `shape[-1] % fmt.block_elements == 0`.
8. Every op supplies `fn work(&self, shapes) -> Work { macs, transcendentals, index_ops, wg_bytes }`. **The verifier rejects a registration whose `work` is a constant.** fusor's `Attention { materialized_bytes: 0, work: 1 }` placeholder cannot recur, and `index_ops` is exactly the term the view-fold-vs-gather tradeoff needs.

**Only Logical can express:** adjoint generation, contraction reassociation, the fold-splitting law, and — the one that earns the level outright — **gradient checkpointing**, because once forward and backward are one Logical graph (§5) "save this activation" versus "recompute it" *is* the extractor's materialization bit.

### 1.2 Launch `nest` — index spaces, kernels, launches

```rust
pub enum Launch {
    Map      { space, body, ops, sched },
    Fold     { space, axis, vec_axes, carrier, acc, post, ops, sched },
    Contract { m, n, k, batch, family, post, acc, a: ContractSide, b: ContractSide, sched },
    Gather   { space, axis, mode, ops, sched },
    Scatter  { space, axis, mode, combine, ops, sched },
    Region   { members: SmallVec<[Id; 8]>, live_outs, sched },
    Ext      { def: OpDefId, ops, attrs: AttrId },
}
```

`Family = Coop | Sgemm | Sgemv`. `ScatterMode = Atomic | SortSegment`. `GatherMode = RowPerGroup | QuantizedRows`. `FoldStrat` (`Subgroup | WgTree{lane_group} | LoopThenTree`) lives inside `ScheduleDomain`.

`Operand { src: Id, layout: Layout, access: AccessPlan }` where `AccessPlan = Alias | Gather | Pack{into} | Unflatten(MultiFlattenMap)`. **Access is an attribute of the edge, not of the producing node** — so one consumer may alias a strided parameter slice while another packs it, which is exactly the trainer's flat-parameter / gradient-concat case (constraint 6) and the im2col operand case coexisting in one graph. `MultiFlattenMap` (one `AxisGroup` per logical axis, each a vector of `SubAxis { extent, stride }` decomposed most-significant-first by divmod, zero strides and colliding strides legal) is carried verbatim; plain per-axis strides cannot express a conv window operand.

`ScheduleDomain` is the enumerable schedule-parameter space of one node — for `Launch::Contract{Coop}` that is `coop_tile_entries() × split_candidates × staging_depth ∈ {1,2}`, ~8,300 points. **It is not e-nodes.** §4 explains how it is resolved without a nested argmin.

**verify_launch.**
1. `Geom::legal(caps)`: `rg*cg*subgroup_width <= max_wg_lanes`; `tm | bm`, `tn | bn`; `bm % (COOP_DIM*rg) == 0`, `bn/n_passes % (COOP_DIM*cg) == 0`.
2. Workgroup footprint is checked against the **exact** value from the shared pure function `arena_plan(tiles(geom, dtype), caps) -> ArenaPlan { total_bytes, placements }`, memoized on `(geom, dtype, caps)` — the *same* function the Kernel emitter uses. There is no estimator, therefore no Launch/Kernel admission mismatch and no "extraction commits a plan that fails Kernel verification and silently falls back" — which would be `hardware_matmul_prep`'s stride-equality bug moved one level down.
3. **A nest's write map must be injective unless the nest declares `combine: Some(c)` with `c.associative`.** One invariant, three jobs: it is the legality rule for scatter-add; it separates the four `Scatter{Add}` lowerings from an illegal in-place write; and applied to a `Window`-derived nest it *is* the proof that a non-overlapping pool's adjoint is an elementwise mask.
4. A `Fold` dim may not appear with nonzero stride in the write map (a fold dim indexing the output is a scatter, not a reduction).
5. Every operand's `AccessPlan` satisfies that operand's access predicate. A failed access analysis disqualifies **this rewrite only**, never every tiled lowering of the expression.
6. A composite node (`Launch::Region`) carries the linear `MapDomain` its members' shared index space implies, checked against the node's *own inferred shape* rather than against whatever the minting rule wrote — so the geometry is a property of the node instead of a field a rule may drift.
7. Every node carries `Effect = Pure | InPlace(BufferRole)`. `Launch::Scatter{Atomic}` and in-place assign are `InPlace`.
8. **Allocation is not described at Launch.** Buffers are derived from the extracted plan (§4).

**Only Launch can express:** fusion, tiling, split-K, layout alias-vs-gather, kernel family, horizontal merging, register tiling, rematerialization.

### 1.3 Kernel `tile` — one kernel body

fusor's `tile-ir` near-verbatim — 14k proven lines with the best verifier in the reference — with five changes: `Shared` is **deleted** (structural sharing comes from hash-consing the whole Kernel term, so two identical subtrees built separately merge, which `Rc::as_ptr` memoization structurally cannot); `AtomicAdd` is added for `ScatterMode::Atomic`; `Stmt::Reduce` is added as the N-ary reduction; `NumericContract` rides on `Unary`/`Binary`; `bf16` joins `ScalarElement`. Element type is runtime data on the node, never a Rust marker or const generic.

**Reduction is N-ary, and the hardware fast path is a degenerate case *beside* it rather than a rewrite of it.** `TileExprKind::Reduce { op: TileReduceOp, kind, value }` survives verbatim as the one-value, one-operator form every subgroup collective and shared-memory tree is spelled with; the single-slot path carries every fold in the system and must keep emitting byte-identical code. `Stmt::Reduce { kind, values, merge: MergeBody, fast, outs, scratch }` is the general form: one partial, one merge expression, one output `Local` and one scratch tile per accumulator **lane** — a `SlotTy::Scalar` slot is one lane, a `SlotTy::Vector(d)` slot is `d`. `fast` is *computed* by the constructor, set exactly when there is one lane whose merge is `binary(op, lhs[0], rhs[0])`, so it can never drift from `merge`; both emitters open their arm with it and take the existing collective path unchanged. **The `accs[0]` bug is unrepresentable**: there is no single `TileReduceOp` to resolve for a whole fold, and `verify_kernel` rejects a node whose `values`, `merge.body`, `merge.lhs`, `merge.rhs`, `outs` and `scratch` disagree in length, so `Fold{(max, sum)}` cannot compute `max(x)` and discard the sum — there is nowhere to discard it to. Cross-lane reads inside `merge` are **required**, not forbidden (flash's running sum and its output accumulator both read the running max); what is rejected is a read of anything outside the formals, because a merge that reads a lane id is not a merge. A multi-lane merge has no hardware collective at all, so it lowers to an explicit log-tree over `lanes * block` scratch with a barrier between levels — and `verify_uniformity` checks the *statement* that produces those barriers, since they are emitted rather than written.

**verify_kernel.** `verify_arena` independently rechecks that every byte-overlapping tile pair is separated by a *guaranteed uniform* barrier, failing lowering rather than racing. A `Barrier` may not appear under an `If` whose predicate is non-uniform over the group. Every `Load` is masked or provably in range; every `Loop` accumulator is declared; every `Stmt::Reduce` is arity- and element-consistent across its lanes with a `merge` reading only its own formals; every `CoopStore` satisfies `cooperative_store_layout_supported`; and expressions are fully type-checked.

**The cut.** Barrier insertion and workgroup arena packing stay **closed-form argmins inside Kernel with an independent verifier**, not e-graph alternatives. Both operate over small tile sets with explicit feasibility predicates. `arena_plan` runs `Regions`, `ByteArena` and the top barrier-insertion candidates and takes the argmin of `total_bytes`. Because `arena_plan` is a pure memoized function of the kernel body and device capabilities, its result feeds `verify_launch` and the Launch occupancy term (`core_workgroup_slots`) exactly, closing the feedback loop without paying e-graph machinery for one occupancy class in one kernel on one vendor.

### 1.4 Deleted levels

The reference has five representations; two carry no optimization. The **execution graph** is a copy of the compute graph with recognizers applied. The **`Operation` trait** is a vtable, not a dialect. Both deleted. Also deleted before being built: an **affine/scf loop dialect** (interchange, unroll, peeling are Launch index-map algebra, decidable, not loop pattern matching) and a **target/WGSL dialect** (naga IR is already an IR; a dialect between Kernel and naga would host zero rewrite rules).

---

## 2. The e-graph

**Cranelift-style acyclic aegraph over hash-consed nodes with union *nodes*.**

```rust
pub struct Node { pub op: Op, pub level: Level, pub children: SmallVec<[Id; 4]> }
pub enum Op { Logical(Logical), Launch(Launch), Union(Id, Id) }
```

**Acyclicity is structural, not checked.** `children` may only contain ids strictly smaller than the node's own id. `union(a, b)` allocates a *new* id `> max(a, b)` holding `Op::Union(a, b)`. There is no union-find, no `rebuild()`, no congruence closure, no cycle probe. The reference's `CAP = 512` bounded reachability probe in region growth — which turns a legal fusion into a rejection non-deterministically with respect to graph size — has nothing to check.

This is the decisive choice over attach-only-with-hash-consing (which has no class merge and therefore silently loses equalities, voiding cross-layer plan sharing) and over union-find-with-rank (where a class's rank becomes a merge artifact and max-rank no longer preserves global acyclicity). Here the invariant is a property of the id allocator, so no rule author can violate it.

**The price, paid deliberately:** equality is not congruent. Unioning `a` and `b` does not union `f(a)` and `f(b)`. Alternatives are minted by rules **at the consumer**. Patterns may match a *spine* — `Views(vs, X)` binds an arbitrary chain of view nodes — which is what makes the reference's self-declared "single clearest structural gap" (`sink_unary_chains_into_matmuls`, impossible there because "a generator may only return a new variant for the node it was asked about") a single-rooted rule here (§3, R5). No multi-root rule form is needed anywhere.

**Sugar and its definition are inserted into the same chain at construction.** `attention(q,k,v,mask)` is not *recognized* from a composite; the macro node and its `defn` expansion are unioned at build time. Recognition ordering, sole-consumer gates and `spike_no_recognition` evaporate — there was never anything to recognize, and the reference's five destructive recognizers with their documented interdependencies do not exist. This also preserves the structural attributes a pattern match would have to re-derive: `MaskKind::Causal` stays on the sugar node (so causality is encoded in the graph and the compiler skips upper-triangle Q·K work without loading a mask tensor) while the decomposition is simultaneously present for algebra and autograd. A `defn` node is never evicted.

**Canonicalization.** `FxHashMap<NodeKey, Id>` hash-consing on canonicalized children. Commutative ops sort children by `Id` at construction, so associativity and commutativity are a **canonical form, not a rule family** — removing the largest blowup source before saturation starts. The global hash-cons replaces, by construction, three reference mechanisms: `Rc::as_ptr`-keyed codegen CSE, `coalesce_equivalent_eclasses`' positional `split_first()` representative choice, and `FusionPlanMemo`'s bounded-depth window whose horizon-completeness was answerable only under `FUSOR_VERIFY_PLAN_SHARING`.

**Budget.** Worklist in creation order; `FixedBitSet` over `(RuleId, Id)`. `MAX_NODES = 8*initial + 4096`, `MAX_ROUNDS = 6`, 2 ms wall. On exhaustion the driver offers only rules tagged `StrictlyLowering`, guaranteeing every chain provably reaches a Launch form — budget exhaustion yields a degraded-but-valid plan, never a hard error. `saturated: bool` and `truncated: Vec<Id>` are reported to conformance; truncation is never silent. Measured target for the trainer's ~3,000-node step graph: 1,900 initial nodes, ~7,400 after saturation, 1.4 ms. The graph stays small because **schedule parameters are not e-nodes** (§4).

---

## 3. The rewrite rule language

```rust
pub struct Rule {
    pub name: &'static str,
    pub level: Level,
    pub head: OpTag,                 // O(1) dispatch filter
    pub tag: RuleTag,                // Additive | StrictlyLowering
    pub apply: fn(&mut Builder<'_>, Id, &Node, &Facts<'_>) -> Option<Id>,
}
```

**`Facts` is a capability token.** It exposes types, shapes, attributes, device caps, `NumericContract` and level invariants. It **structurally does not expose** consumer counts, liveness, cost, or extraction state. Guards therefore encode **legality only, never profitability** — and that is enforced by the type system rather than by convention. This single restriction is the design's immune system: fusor's `consumer_count(input) != 1`, `skip_externally_live`, `variant_duplicates_required_producer` and `merge_profile -> None` are all profitability judgements smuggled into legality gates, which is why two individually profitable optimizations end up jointly illegal with neither knowing. Here profitability lives in the cost model or nowhere. **Rule order carries no semantics**; the fixed order exists only for reproducibility.

Two syntactic forms: a `rule!` `macro_rules!` for structural patterns, and a plain `fn` for rules that do arithmetic. No proc macro — four of the ten interesting rules enumerate integer tuples and a pattern DSL would not earn itself.

**R1 — algebraic: the fold-splitting law.** The reference's `fold.rs` names the law that split-K, online softmax and Welford are one fact, then implements all three separately.

```rust
rule! { fold_split, Logical, tag = Additive,
    Fold { carrier: c, axis: a, acc, ins: [x] }
      if c.associative && facts.own().numeric.reassoc && dim(x, a).at_least(4096) =>
    Fold { carrier: c.as_merge(), axis: a, acc,
           ins: [Fold { carrier: c, axis: a + 1, acc,
                        ins: [Restride::block(x, a, blocks: Sym::fresh())] }] } }
```

At a `Contract`'s summed axis it *is* split-K; at a `(max, sum)` carrier it *is* online softmax; at `(n, mean, m2)` it *is* the stable variance accumulator. The outer level reads partial **accumulators**, hence `as_merge` — reusing the inner carrier applies `lift` to a partial max and silently computes a wrong value, which at a single-slot binop is invisible. The `reassoc` guard is not decoration: without it this rule declares the split and unsplit forms **value-equal**, and extraction swaps them on cost, on f16 accumulators, in a system whose acceptance test is a byte-identical QAT export.

**R2 — fusion: producer inlining, which is also region formation.**

```rust
rule! { map_into_fold, Launch, tag = Additive,
    Launch::Fold { space, carrier, post, acc, x: Launch::Map { body, ops, .. } }
      if space.covers(x.space) && ops.iter().all(|o| o.access.legal_in(space)) =>
    Launch::Fold { space, carrier: carrier.with_lift(carrier.lift.compose(body)), post, acc, ops } }
```

No consumer-count check. No duplication veto. If `x` has two consumers, both may inline it and the **cost model** charges the recompute once per consumer against the saved write plus the saved reads — exactly what the reference's `spike_dup_ledger` measures and then discards. `Launch::Region` is the same rewrite with `live_outs` non-empty: a multi-output node emitting an extra buffer. The reference's contradiction — the extractor vetoing precisely the fusions `form_elementwise_regions` then performs by a different rule — is unstateable.

Note also that elementwise-into-elementwise fusion needs **no rule at all**: it is `pre.compose(body)`, a `ScalarExpr` tree substitution inside `Map`.

**R3 — tiling/layout: the domain rides on the family mint.**

```rust
fn lower_family(b: &mut Builder, id: Id, n: &Node, f: &Facts, family: Family) -> Option<Id> {
    let dom = CoopDomain::legal(n, f.caps());      // filters by arena_plan bytes + lane limits
    if dom.is_empty() { return None }
    b.union(id, Launch::Launch::Contract { family, sched: ScheduleDomain::Coop(dom), .. })
}
```

A separate `tile_contract` rule that upgraded a `ScheduleDomain::Point`
contraction used to sit here. Nothing ever minted one: `lower_floor.rs` is the
only source of `ScheduleDomain::Point` and it mints no `Launch::Contract`, so the rule
matched zero nodes over the whole conformance suite and every model, and it is
deleted. The domain is attached where the family is chosen — the same "replaces
`Point` instead of competing with it" rule `tile_gather`'s note states.

**One node, not four and not four hundred.** The full legal `(geom × splits × staging_depth)` space is carried on the node and resolved by extraction (§4). This is the resolution of the sharpest disagreement in the panel: minting every point blows the graph to ~90k nodes on a 32-layer transformer; minting a `score_fs`-Pareto top-4 makes a cheap local heuristic *gate* the real cost model, so the true winner may never be minted (the reference's `score_fs` cannot see an epilogue's downstream traffic); and a nested argmin *inside* the node's cost function is circular, because the geometry it picks determines the output's padded strides, which determine every consumer's read traffic and materialization demand. Carrying the complete domain and resolving it as a **move in the global search** is the only formulation that is simultaneously small, complete and non-circular.

`SgemmDomain::legal` and `SgemvDomain::legal` *generate* every `(BM,BN,BK,TM,TN)` / `(chunk,vector,subgroups)` tuple satisfying the structural predicates `fallback_family_params_are_legal_everywhere` already asserts. The 200-line SGEMM regression tree and the 21-arm SGEMV bucket table are deleted; their measured leaves seed move ordering only.

**R4 — kernel selection: four independent, order-free rules.**

```rust
rule! { lower_coop,    Logical, tag = StrictlyLowering, Contract{..} if f.caps.coop_supported
        => Launch::Contract { family: Coop,    sched: ScheduleDomain::Coop(..), .. } }
rule! { lower_sgemm,   Logical, tag = StrictlyLowering, Contract{..}
        => Launch::Contract { family: Sgemm,   sched: .. } }
rule! { lower_sgemv,   Logical, tag = StrictlyLowering, Contract{..}
        => Launch::Contract { family: Sgemv,   sched: .. } }
rule! { lower_generic, Logical, tag = StrictlyLowering, Contract{spec, acc, a, b}
        => Launch::Fold { carrier: Carrier::binop(Add).with_lift(mul(Arg(0), Arg(1))), acc, .. } }
```

`ShapeSelector`'s first-match ordering — in which a gemv-shaped contraction picks Coop, the tile scorer declines on its padding gate, and production silently runs a third path (the reference pins this as golden row `1x4096x4096 => Coop tile=None`) — is structurally impossible: all four coexist. The `padded_macs * 4 > useful_macs * 5` routing guard is **deleted**, because padded MACs already enter the issue term, so a badly padded coop tile simply loses to sgemv on cost instead of being routed around it. The nine `qmatmul_direct_selector` arms become nine independent rules with divisibility as a legality guard and `N >= 8192` / `K <= 1024` as cost terms, so N=8191 degrades continuously.

**R5 — sinking, consumer-rooted.**

```rust
rule! { sink_epilogue, Launch, tag = Additive,
    Launch::Map { body, ops: [Operand { src: v, access: Alias, .. }] }
      if let Some((mm, views)) = b.trace_pure_views(v) =>
    Restride { specs: views, x: Launch::Contract { post: post_of(mm).compose(body), ..mm } } }
```

**R6 — scatter-add, four lowerings.** `Atomic` (guarded on `caps.atomic_f32`), `SortSegment`, `WgPrivateMerge` (guarded on `rows*elem_bytes <= caps.max_wg_storage`), and `OneHotContract` — the last surviving only as the candidate the cost model rejects. At the trainer's batch-128 / 768-unit / K=3 shape, `OneHotContract` prices at 1.2 GB of traffic against `WgPrivateMerge`'s 96 KB private accumulator (1024 bins × 24 f32, fits threadgroup memory). The trainer's host-side three-level sorted gather-and-sum, its `ScatterShape` padding and its ~0.9 ms/batch host cost are all deleted.

**R7 — `specialize_dim`** substitutes a `Sym` for its concrete binding, priced by compile amortization (§4). **R8 — `qrepack`** mints `Operand { layout: F32Scales }` from `Native`, cost = the repack bytes amortized over `Persistence::Persistent`.

---

## 4. Cost model and extraction

**One scalar, picoseconds, on a roofline.** Not a lexicographic tuple: the reference's own unit test (`scalar_cost_lets_traffic_outweigh_a_dispatch`) shows the tuple gives the wrong verdict, and its own doc concedes dispatches are 0.2% of modelled time while the tuple will pay unbounded bandwidth to remove one.

`DeviceFacts` carries `launch_ps`, `dram_bytes_per_us`, `llc_bytes`, `wg_bytes_per_us`, `mac_per_us[Fma|Coop|Dp4a][dtype]`, `trans_ps`, `store_ps_per_element`, `saturation_lanes`, `single_buffered_traffic_pct`, `compile_ps_per_kernel`, plus the wgpu limits. `fusor-cost::facts::seed_facts(&Caps)` builds them from the capabilities the backend reports, per device *class* and in physical units. This closes the reference's largest portability liability: five integers fitted on one M2 Max, selected by `backend == Metal && name.starts_with("Apple")`, shared by every other GPU on earth.

Per launch: `launch_ps + max(dram_ps, math_ps, wg_ps) + drain_ps`.

- `dram_ps` counts **reads and writes**. An operand reread `r` times costs `bytes` when `bytes <= llc_bytes` and `r*bytes` otherwise, interpolated — a continuous LLC term, not the reference's strict `>` cliff where one byte over 8 MiB flips the tiling plan. Grid swizzle is a term reading `llc_bytes` from the one device-fact source, deleting the private `LLC_CLASS`/`SMALL_B` constants.
- `math_ps = macs/mac_per_us + trans_ops*trans_ps + index_ops/mac_per_us`, scaled by the occupancy shortfall `max(1, saturation_lanes/launched_lanes)^(1/3)`.
- `wg_ps` and `drain_ps` are `score_fs`'s T2 and T3, with residency from `arena_plan().total_bytes`.

`score_fs` maps onto this term-for-term (T1→math, T2→wg, T3→drain, T4→the `max`, T5→the combine launch); its measured anchors ship as the seed calibration for the Apple class. **Precision is not a cost term** — it is a verifier property (§1.1), because a time-only model eliminates f32 everywhere.

**Compile amortization.** `compile_ps_per_kernel / expected_reuse(plan_hash, binding)`, with `expected_reuse` from a bounded per-process `ShapeStats: FxHashMap<PlanHash, SmallVec<[(DimVec, u32); 8]>>`. On first sighting of a shape family the generic symbolic variant wins outright — nothing compiles per length bucket. After a binding recurs, `specialize_dim`'s variant wins where specialization pays. The trainer's ten sequence buckets, its `tiles = slots.div_ceil(64) + 1024` padding and `--bench`'s single-bucket filter all become unnecessary, and specialization is a decision recorded in the key rather than an accident of shape.

### 4.1 Compositionality: the shared-subexpression trap

Naive extraction assumes `cost(n) = local(n) + Σ cost(children)`. That is wrong twice for tensor programs: a shared value is counted once per path (or once globally, both wrong), and *materialization* is not a property of any node.

**Cost is therefore not defined per e-class.** Extraction state is a triple:

```rust
pub struct Extraction {
    sigma:   FxHashMap<Id, Id>,   // union-chain root -> selected node
    m:       FixedBitSet,         // materialized set
    theta:   FxHashMap<Id, SchedPoint>,   // schedule point per selected ScheduleDomain node
}
```

Cost is evaluated on the **realized DAG** under `(sigma, m, theta)`:
- A node in `M` pays one write of its bytes; each consumer pays one read.
- A node not in `M` is inlined into every consumer: it pays its `math_ps` **once per consumer** and no traffic.
- Launches are the connected components of the realized DAG cut at `M` boundaries and at forced boundaries (index-space mismatch, fold-to-fold dependency).

Consumer counts come from the DAG, so **rematerialization is priced exactly** as `saved_write + saved_reads - recompute*(consumers-1)`. The reference's `variant_duplicates_required_producer` hard veto — which makes the whole remat space unreachable — is deleted, not weakened.

**Effect pinning.** A selected node with `Effect::InPlace` is **pinned in `M`**. Without this, toggling a two-consumer `Launch::Scatter{Atomic}` out of `M` inlines it into both consumers' kernels and the atomics apply twice, doubling the embedding gradient. Purity is thus a precondition of the materialization move, not an afterthought.

### 4.2 The algorithm

1. **Admissible lower bound**, bottom-up, `O(nodes)`: `lb(c) = min over n in c of ( math_ps(n) + Σ over *distinct* child chains lb(child) )` — zero traffic, free sharing, min over the schedule domain. It is a genuine relaxation in both regimes: an inlined node's true cost pays math `k` times where `lb` pays once; a materialized node's true cost pays math plus traffic where `lb` pays math. Admissible, so it is a valid seed *and* a valid branch-and-bound prune. (The alternative seed — assume everything shared is materialized — is not a lower bound at all; it maximizes launch count and pays a write plus a read for every edge the optimal fused cut deletes, which is precisely the conv-epilogue shape the trainer is made of.)
2. **Seed** `sigma_0 = argmin lb`; realize; `m_0 = roots ∪ {c : consumers(c) > 1} ∪ {c : index-space mismatch} ∪ {c : Effect::InPlace}`; `theta_0` from the local `score_fs` ranking.
3. **Exact cost** on the realized DAG.
4. **Local search** over three moves: `RESELECT(c)` (change node), `FLIP(c)` (toggle `M`, refused when pinned), `RESCHEDULE(c)` (change schedule point). Each move's delta is recomputed over only the affected launches via a union-find over the realized cut. Accept strict improvements; ties by node id. `RESCHEDULE`'s move frontier is ordered by `score_fs` (Pareto top-k first) but **the full domain remains reachable and the accept test is always the exact global cost** — `score_fs` orders moves, it never gates candidates. Schedule cost is memoized on `(node, theta, context_hash)` where `context_hash` covers segment count, epilogue signature, operand layouts and the consumer demand set, which is what makes an ~8,300-point domain affordable.
5. **Budget** `64*|chains|` moves or 2 ms, keeping best-so-far. Fully deterministic.
6. **`verify_plan`** on the winner: every selected non-leaf is Launch; every geometry legal against the exact `arena_plan`; every operand access satisfiable; every buffer stride derivable; no `InPlace` node inlined. A failure is a **hard conformance assert**, never a silent fallback.

**Allocation is derived from the plan**: for each node in `M`, `buffer_layout(node, theta)` gives the padded strides the selected geometry needs, including split-K scratch slices. `hardware_matmul_prep`'s exact-stride equality test and its silent generic-reduce fallback become an invariant the extractor establishes.

**The plan is the cache key.** `PlanHash = hash(realized DAG term + M + theta + DeviceFacts::fingerprint)`, where the fingerprint **includes `max_compute_workgroup_storage_size`** (the reference omits it while the coop legality filter reads it — a live staleness hazard). `Dim::Sym` and `Leaf::Uniform` hash as symbols, not values, so one plan serves a whole shape family. There is no `hash_kernel_fields`, no `kernel_cache_key_with_dispatch`, no `structural_kernel_key`, no golden byte files; the class of bug where a new decision variable must be threaded into four hash recipes cannot exist.

**Replay.** The extraction memo is keyed on `(Logical term hash, DeviceFacts fingerprint, dim binding vector)`. On a miss we re-extract (~1.4 ms) and compare `PlanHash`; unchanged means nothing recompiles. Validity is *"the extraction inputs are identical"*, not *"a structural fingerprint matches"*, so a training loop can no longer freeze step 1's decisions forever. This is affordable precisely because the trainer reads nothing back and the host runs several steps ahead — re-extraction never lands on the critical path.

**Verification of the heuristic.** `fusor-conformance` ships a **debug ILP oracle** that must agree with the local search on small graphs, plus golden `PlanHash` tests over the calibration shape set and a `--exhaustive` mode. A greedy search compared only against itself cannot distinguish "found the optimum" from "made the same mistake as the reference."

---

## 5. Autograd

**An Logical → Logical transform in `fusor-autograd`, run before ingestion, whose output is ingested together with the forward as one graph with one root set** (parameter gradients plus any requested loss).

*Why Logical*: adjoints are facts about tensor algebra. `d(Contract) = (grad @ Bᵀ, Aᵀ @ grad)` holds regardless of tile geometry; stating it at Launch would mean restating it per lowering.
*Why not as rewrite rules*: an adjoint is a directed transformation, not an equality; putting `grad` in the primal's chain is unsound.
*Why one graph*: **gradient checkpointing is the extractor's `M` decision.** An activation alive for the backward is a node in `M`; recomputing it is dropping it. Nobody writes a checkpointing pass and there is no user annotation. `Persistence` tells the extractor which buffers it may recompute.

```rust
pub enum AdjointKind {
    Analytic(fn(&mut Tape, &Node, Val, &[Val], Val) -> SmallVec<[Option<Val>; 4]>),
    Structural,        // derived from the op's own attributes
}
pub static ADJOINTS: &[Adjoint] = &[
    Adjoint { op: Contract, kind: Analytic(|t,n,g,i,_| smallvec![
        Some(t.contract(g, i[1], n.spec().d_lhs())),
        Some(t.contract(i[0], g, n.spec().d_rhs()))]) },
    Adjoint { op: Map,      kind: Analytic(map_adjoint) },
    Adjoint { op: Restride, kind: Structural },
    Adjoint { op: Window,   kind: Structural },
    Adjoint { op: Gather,   kind: Structural },
    Adjoint { op: Scatter,  kind: Structural },
    Adjoint { op: Fold,     kind: Structural },
];
```

**Seven entries.** `map_adjoint` differentiates a `ScalarExpr` once and thereby covers all 23 unaries, all 12 comparisons (which differentiate to zero automatically, satisfying the invariant that every requires-grad parent receives a gradient), `where_cond`, `clamp`, `gelu`, `sigmoid`, `silu`, and the scalar-arith family. `Fold`'s structural adjoint reads the carrier's own **merge**, not a variant name: a single scalar slot merged by `Add` broadcasts, `Mul` is the zero-aware product rule, `Max`/`Min` use the declared `Carrier::tie` (`SplitEvenly | FirstWins`) rather than an implicit even split. A multi-slot carrier has no analytic adjoint and says so — those carriers are minted by the fold laws *after* autograd, and the composed adjoint of the chain they replaced is what carries the gradient.

**`Window`'s structural adjoint is trainer constraint 4, solved by two integers.** From `(window, step)`: `step >= window` proves the adjoint is an elementwise mask-and-broadcast; overlapping windows give `Scatter{Add}`, itself a chain with four lowerings. This is why `Window` survives as a core op rather than collapsing into `Restride` with injectivity as a derived fact: under `Dim::Sym`, injectivity of a *relative* stride composition with symbolic extents and offsets is undecidable, the verifier must answer conservatively, and the trainer's non-overlapping max-pool would degrade to a scatter on exactly the symbolic-shape path the design exists to enable.

**There is no `replay_*`.** `conv`, `grouped_conv`, `rms_norm`, `layer_norm`, `rope`, `attention`, `upsample`, `pool`, `q_mat_mul` are macro ops whose `defn` expansion into core Logical is present from node zero, so their adjoints are the composition of core adjoints, automatically. The reference's four replay combinators — build a throwaway `Graph` at backward time and re-differentiate — do not exist. The façade's autograd tape records leaves and explicit user chain-rule boundaries; it does not store every forward op as an `Arc<dyn Fn>`, downcast type-erased tensor values, or create self-`Arc` cycles.

The named risk is that the reference's **hand-fused backwards** (`attention_grads`, norm backward, the analytic softmax Jacobian) must be recovered from the composed backward. Conformance value-checks the named backward shapes so a broken rewrite is a hard failure rather than a quiet throughput regression. The trainer's `distillation_loss` becomes the plain taped softplus chain, with `softplus_bce_adjoint` rewriting its adjoint to the single-sigmoid form; QAT fake-quant is one `with_backwards` registration with zero user code. `with_backwards` survives as the escape hatch, using `Parent` and a bare-node `GradientSlot` so a closure cannot close an `Arc` cycle, plus validation that every requires-grad parent receives a gradient.

---

## 6. Both backends

**Shared:** Logical, Launch, Kernel, the e-graph, every Logical rule, autograd, the cost-model *shape*, the quantized decode tables, `arena_plan`, the plan cache, and the tile dialect itself. A WGSL `vec4`/`fma`/`dot` and a CPU `f32x4::mul_add` are the same primitive at different widths.

**Target-specific:** `DeviceFacts` values, `Caps`, the Launch lowering rules that mention lane/subgroup geometry, and the Kernel emitter.

```rust
pub trait Target: Send + Sync {
    fn caps(&self) -> &Caps;
    fn facts(&self) -> &DeviceFacts;
    fn rules(&self) -> &'static [Rule];
    fn emit(&self, ir: &KernelIr) -> Result<Artifact, EmitError>;
    fn launch(&self, a: &Artifact, grid: [u32;3], binds: &[Buf], uni: &Uniforms) -> Result<()>;
}
```

**GPU.** Kernel → liveness → `arena_plan` → `verify_arena` → naga `Module` → validate → `create_shader_module_trusted`. Bind groups stay **derived** from the emitted module's storage globals, sorted by binding, read-only from the absence of `StorageAccess::STORE`, zipped positionally with the builder's buffer list — binding order and codegen cannot drift. One `main`, one bind group, whole-buffer bindings. **Binding 0 is always a `Uniforms` storage buffer** holding `[u32 symbolic dims..., f32 uniform scalars...]`. That one buffer kills trainer constraints 1 and 2 together: `m * lr_f32` produces a `Uniform`, not a baked literal, and a sequence length is a `Sym` read from binding 0. (A separate uniform-address-space block would break the derived-bind-group mechanism, which walks storage globals; that is why this is a storage buffer.)

Default build targets released **wgpu 29.x** with `naga-ir` and requests **WebGPU baseline limits**, widening only what a selected kernel's legality predicate proves it needs — so a plan legal on one device is legal on another and the cost model's filters mean the same thing everywhere. SUBGROUP, SHADER_F16, EXPERIMENTAL_COOPERATIVE_MATRIX, PIPELINE_CACHE and TIMESTAMP_QUERY are each probed with a working fallback (`WgTree` folds, f32, `Family::Sgemm`, cold compile, no profiling). The wgpu fork is one cargo feature, `fork-metal`, contributing exactly two capabilities — workgroup-alias byte-arena packing and mixed-precision cooperative store. Without it the arena falls back to per-stride-class regions and an f32-accumulated f16-output kernel pays a staging tile plus a per-lane cast: footprint and a staging pass, never correctness.

Buffers come from a pooled allocator keyed `(size, usage)` with `strong_count == 1` reuse and a platform memory ceiling that blocks and retries before failing (on macOS, exceeding unified memory kills the OS rather than erroring). Host syncs are exactly: explicit readback, explicit wait, and the allocator's cap retry — back-pressure on in-flight submissions is a runtime policy, not a `--drain-every` counter in the training script.

**CPU.** The same `KernelIr`, different emitter:

| Kernel concept | GPU | CPU |
|---|---|---|
| `grid` | `dispatch_workgroups` | `parallel_for(0..grid, grain)` on one persistent pool |
| `block` lanes | invocations | `W` SIMD lanes × unroll; `W` from a cached `Level` |
| workgroup tile | `var<workgroup>` | thread-local 64-byte-aligned scratch |
| `Reduce{Subgroup}` | subgroup collective | horizontal reduce over `f32x8` |
| `Reduce{WgTree}` | shared-memory tree | tree over the scratch tile |
| divergent `If` | `if` | `select` on a lane mask |
| **`Barrier`** | `controlBarrier` | **splits the lane loop into two loops over the lane range** |
| `AtomicAdd` | `atomicAdd` | per-thread private accumulate + merge |

The barrier mapping is not cosmetic. A block's 256 lanes become an inner loop of 32 iterations at `W = 8`; mapping `Barrier` to a no-op miscompiles every kernel that stages through workgroup memory — `Reduce{WgTree}`, packed-B GEMM tiles, and the flash-attention staging step — because iteration 0 reads tile slots iteration 31 has not written. Splitting the loop is the correct semantics, costs one lowering pass, and makes the arena separation predicate trivially true on CPU.

`fearless_simd`'s statically-known `f32x4/f32x8/f32x16` is what makes a 4×4 register accumulator tile expressible at all; `pulp`'s width-erased `S::f32s` structurally cannot, which is the root cause of the reference's `[E; 64]` spill pattern in every comparison, every transcendental and every strided gather. `Level` is cached in a `OnceLock` and dispatched **once per kernel launch**, not per row. `pulp`, `gemm` and transitive `rayon` are dropped: an external BLAS in the critical path makes epilogue fusion structurally impossible. Matmul is a `Launch::Contract` lowered to Kernel with real blocking, packing and a microkernel, so bias/gelu/dequant epilogues fuse into the k-loop. f16 and bf16 compute is the `widen-compute` **lowering rule** (widen to f32 registers, compute, narrow on store), not a one-lane `F16Scalar`. Non-contiguous access is four lowering alternatives — contiguous, broadcast/splat, unit-inner-stride sub-slice, general gather — plus a `Pack` operand access, never a per-vector runtime `is_contiguous()` branch. Parallelism is a scheduling attribute on an outer Launch tile loop priced against real pool-wake cost, deleting `PARALLEL_THRESHOLD = 16_777_216`.

**A third backend** implements `Target` (~1,500 lines: facts, caps, emitter, launcher) and optionally contributes Launch rules. It inherits every Logical rule, autograd, the cost model and the plan cache for free.

---

## 7. Quantization and mixed precision

**Formats are data.**

```rust
pub struct BlockSpec {
    pub elements: u16, pub bytes: u16,
    pub decode: BlockProgram,           // a small Kernel snippet, not a single ScalarExpr
    pub native_f16_scales: bool,
    pub activation: &'static [QAct],
}
```

`decode` is a **block program**, not one scalar expression: Q6K's 210-byte non-word-aligned block with per-super-block group scales is not a per-element formula, which is exactly why the reference needs `Q4KBlockParts`/`Q6KBlockParts`. Adding Q4_1 is a table row plus a block program — not a kernel and not a selector arm. The six ingestible formats (Q4_0, Q5_0, Q8_0, Q4K, Q5K, Q6K) plus F16/F32 passthrough are the parity requirement; the twelve named-but-unreachable formats in the reference's enum do not appear.

**Storage layout is an operand attribute; repacking is a priced rule.** Both `Native` and `F32Scales` are legal inputs everywhere, and `qrepack` costs the repack bytes amortized over `Persistence::Persistent`. A device that never runs the alignment-sensitive `qgemv_q6k_ggml` kernel no longer pays +0.9% bytes to satisfy that kernel's addressing requirement, and layout no longer feeds back into routing through `Q4_0` vs `Q4_0Native` format variants.

**Both quantized dot lowerings coexist**: `QAct::F32` (dequantize the block into registers, then FMA) and `QAct::Q8Dp4a` (`Pack4xI8Clamp` the activations, `Dot4I8Packed` against still-quantized weights — provably not expressible as dequantize-then-dot). The cost model chooses; the reference's hardcoded comment about compounding error becomes a `NumericContract` guard on the int path.

**Mixed precision is an accumulator attribute.** `Launch::Fold` and `Launch::Contract` carry `acc: Dtype` separately from operand dtype, so `contract{acc: F32}(F16, F16) -> F16` is one node and `coop_epilogues_supported` reads `acc` rather than inferring from a chain. When the coop kernel cannot host an epilogue, un-fusing it into a second dispatch is **one alternative** and routing to the generic fold is another — never the only outcome. `cast` is a `ScalarExpr` node inside `Map`, differentiable both directions by `map_adjoint` with no special case, so the trainer's f32-master / f16-conv-stack recipe and its 1024× loss scale are ordinary graph structure.

---

## 8. Crates and the user-facing API

```
fusor-ir          Levels, Dim, Dtype, ScalarExpr, Node/Egraph, Rule registry, OpDef registry,
                   verify_l0 / verify_launch                        (no wgpu, no SIMD, no device)
fusor-tile        Kernel dialect, liveness, arena_plan, verify_arena, uniformity      -> ir
fusor-autograd    ADJOINTS table, Logical->Logical transform                                -> ir
fusor-cost        DeviceFacts, calibrate(), cost(), extraction, ShapeStats        -> ir
fusor-gguf        BlockSpec tables, GGUF parsing, VarBuilder (sync/sharded/async)
fusor-gpu         wgpu Target, naga emitter, GPU rules, BufferPool, plan cache    -> ir,tile,cost,gguf
fusor-cpu         fearless_simd Target, SIMD emitter, CPU rules, worker pool      -> ir,tile,cost,gguf
fusor             Tensor, ops, macro ops, layers, Graph, optimizers, Session      -> all
fusor-conformance op x backward matrix, resolves_in asserts, ILP oracle, --exhaustive
```

Core ops are closed enums (`Logical`, `Launch`) so the verifier, cost model and emitters match exhaustively. Extension ops enter through the single escape `Launch::Ext { def: OpDefId, .. }` against an open `OpDef { tag, verify, infer, work, adjoint, lower_per_target }` registry. That is one uniform extension shape, and no core file changes to add an op.

```rust
let s = Session::new(Device::gpu().await?);
let g = Graph::new(&s);
let w = g.param("w", [out, inp], Dtype::F32);
let b = g.param("b", [out], Dtype::F32);
let y = x.matmul_t(&w).add(&b).gelu();     // x: [Sym("rows"), inp]
```

The public root is `Tensor<const R: usize, T = f32>`, a transparent const-rank façade over `tensor::Dyn`. `Dyn` remains the runtime-rank, fallible escape hatch for loaders and heterogeneous passes; neither type affects the IR. There is no `B: Fusion<R, D>`, `OUT_RANK`/`DIFF`/`MaxRank` witness family, rank ceiling of 21, or separate `Typed` alias. Compatibility aliases that duplicated the natural spelling were removed; model-facing operations live as inherent `Tensor` methods.

The Logical that produces:

```
%3 = contract %x, %w {m:[s0] n:[64] k:[24]} acc=f32       : f32[s0, 64]
%4 = restride %b [{d:0,mult:0,size:s0},{d:0,mult:1,size:64}] : f32[s0, 64]
%5 = map add(Arg0, Arg1) (%3, %4)                          : f32[s0, 64]
%6 = { map gelu_tanh(Arg0) (%5)            ← defn, minted at construction
     , macro gelu %5 }                     ← sugar, same chain
```

After saturation `%6`'s chain also holds `Launch::Contract{Coop, sched: <8,300 points>, post: add(bcast %b) . gelu}`, `Launch::Contract{Sgemv, ..}`, `Launch::Contract{Sgemm, ..}`, `Launch::Fold{Add, ..}`, and `Launch::Map` reading a materialized `%3`. At `s0 = 4096` on an M2 Max, extraction selects the Coop node, resolves `theta` to a concrete `(geom, splits, depth)`, inlines `%3` and `%5`, aliases `b`, and emits **one launch**. On a device without cooperative matrices the same graph and the same rules with different `caps` select Sgemm. Nothing in the frontend changed.

---

## 9. Minimality: what is not in the core

The core is **ten Logical nodes over one closed scalar vocabulary**, one Launch op family, one tile dialect, one cost function, one extraction algorithm, one rule table.

| Not in core | Mechanism |
|---|---|
| 60 elementwise opcodes, 12 comparisons, `where_cond`, `clamp`, all activations | one `Map` with a different `ScalarExpr` |
| softmax ×5, rms_norm ×4, layer_norm ×2, var, mean | macro + `defn` → `Fold`+`Map`; refused into one launch by `fold_split` + `map_into_fold` |
| conv, grouped_conv, pool_max/min, upsample | macro + `defn` → `Window` + `Contract` / `Window` + `Fold` |
| attention (+lse, +grads, +causal) | macro carrying `MaskKind` + `defn`; the expansion competes through ordinary Launch alternatives |
| matmul, mat_mul_transposed_rhs, q_mat_mul | `Contract` with an `EinSpec`; transposed-rhs is a spec, not an op |
| cat, stack, pad_axis, repeat, slice_assign | `Scatter{Set}` into `Leaf(Const)` |
| index_select, embedding, gather_last, `i()` | `Gather`; adjoint is `Scatter{Add}` |
| ~22 view ops | `Restride` |
| rope ×6, RopeCache | macros whose expansions can fuse into ordinary Launch regions |
| five destructive recognizers and their fixed order | `defn` at construction — there is nothing to recognize |
| `MatMulParams`, `ShapeSelector`, SGEMM tree, SGEMV table | four order-free lowering rules + candidate generators; family is never stored on a node |
| sink_unary_chains, form_elementwise_regions, alloc_reuse | `sink_epilogue`, `map_into_fold`/`Launch::Region`, plan-derived allocation |
| merge_horizontal | **not carried over.** A `KMerged` wave node and three merge rules existed; over the conformance suite, a decode and all four model crates the mint guard refused every candidate (74,488 attempts, 0 waves) and nothing was ever lowered, so the node, its rules and its two lowerings were deleted. Horizontal fusion is available through `Launch::Region`. |
| split-K, online softmax, stable variance, log-sum-exp | one `Carrier` + one `fold_split` rule; no combine is named in the core |
| DispatchPolicy, all 8 `spike_*` flags, `PARALLEL_THRESHOLD`, `graph_flush_threshold` | named `DeviceFacts` fields with a calibration path; per-shape cost terms; measured-better default shipped |
| 21 rank-witness traits, `ShapeWithOneHole`, `B: Fusion` | const-rank `Tensor<R, T>` façade over runtime-rank `Dyn`; rank changes use ordinary const parameters and `verify_l0` |
| `Operation` trait, `hash_kernel_fields`, 3 key recipes, `key_goldens` | `PlanHash` over the extracted term |
| replay combinators ×4, the mutable tape | `defn` makes composite adjoints compose; 7 `ADJOINTS` entries |
| top_k, standard/mirostat-2 samplers | `Launch::Ext` + `OpDef`: inference-only, no adjoint, one declared cost row |
| barrier insertion, arena packing | closed-form argmins inside `arena_plan`, exact-valued into `verify_launch` and the occupancy term |

The last two rows are honest exclusions rather than reductions, and are marked as such.

---

## 10. How to add

**An op.** Write a macro-op constructor that mints the sugar node and unions its `defn` expansion in the same call. If it needs a fused kernel, add one `Rule` minting an `Launch` node plus a `fn(&Launch) -> KernelIr` per target and a `work()` row. If its adjoint is not the composition of core adjoints, add one `ADJOINTS` entry. Nothing in `fusor-ir` changes.

**An optimization.** Write a `Rule`, add it to the owning crate's `RULES: &[Rule]`, add a conformance case asserting it fires and one asserting extraction selects it on a named shape. Guards may read only `Facts`; profitability goes in the cost model. No scheduler edit, no phase to insert into, no cache-key threading.

**A backend.** Implement `Target`: `caps`, `facts` (plus a `calibrate` run), `rules` for target-exclusive lowerings, `emit`, `launch`. Everything from Logical through Kernel including autograd, the cost model and the plan cache is inherited.

**A dtype.** Add a `Dtype` variant, a `ScalarElement` row per emitter, cast entries in the `ScalarExpr` table, `mac_per_us` rows in `DeviceFacts`, and — if it is a storage-only type — one `widen-compute` lowering rule. Quantized formats add a `BlockSpec` row instead.

---

## 11. Delivery

- **M0 (wk 1)** `fusor-ir` + `verify_l0` + `fusor-cpu` with one trivial lowering rule per Logical op. No e-graph, no tiling. Correct and slow.
- **M1 (wk 2)** `fusor-autograd`, seven adjoints. `train_xor` passes on CPU.
- **M2 (wk 3)** `fusor-gpu`, same trivial rules. **The trainer trains end-to-end on Metal, slower than the reference.** This is the architecture acceptance gate, and it lands before any optimization exists — the only empirical test of a minimality claim is a running training step built from the core alone.
- **M3 (wk 4)** e-graph + cost model + extraction with fusion rules only. Launch count collapses to reference parity.
- **M4 (wks 5-6)** `tile_contract`'s `ScheduleDomain`, coop lowering, `score_fs` port, `calibrate`. Matmul parity.
- **M5 (wk 7)** scatter lowerings, the `Window` structural adjoint, `specialize_dim`, and attention-region lowering. All four trainer workarounds deleted; MSQ1 byte-identical; parity and throughput gates enforced in CI.
