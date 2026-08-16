# fusor2 — archived scaffold contracts

This records the original scaffold contract for the fourteen implementation
work items. It is historical: module visibility and façade names have changed
since those work items landed, so the signatures below are not a supported API
inventory. See [`API.md`](./API.md) for the current public façade and the crate
roots for current internal seams.

Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the current design rationale.

---

## 0. Ground rules

- **Ownership.** You own exactly the files listed for your work item. `Cargo.toml`,
  `lib.rs` and every `mod.rs`-equivalent (`semantics.rs`, `rules.rs`, `ops.rs`,
  `domains.rs`, `emit.rs`, `lower.rs`, `suite.rs`, `cache.rs`, `layers.rs`,
  `composite.rs`, `sampling.rs`, `tensor.rs`) belong to the scaffold. If you need
  a **new** file, say so in your return value rather than editing a module list
  another agent may also be editing.
- **No `todo!()`/`unimplemented!()` in anything you claim is done.** Every stub
  body you leave behind must be reported.
- **Edition 2024**, `rustc` 1.88+. Workspace deps are pinned once in the root
  `Cargo.toml`; use `foo.workspace = true`, never a fresh version.
- `cargo check --workspace --all-targets` passes as of the scaffold. Keep it
  passing.

---

## 1. Crate graph

```
fusor2-ir          (no deps)
fusor2-tile        -> ir
fusor2-autograd    -> ir
fusor2-cost        -> ir
fusor2-gguf        -> ir
fusor2-gpu         -> ir, tile, cost, gguf
fusor2-cpu         -> ir, tile, cost, gguf
fusor2             -> all of the above
fusor2-conformance -> all of the above
```

Shared external deps, pinned once: `smallvec 1.15` (feature `union`),
`rustc-hash 2.1`, `fixedbitset 0.5`, `half 2.6`, `bytemuck 1.25`,
`parking_lot 0.12`, `lru 0.14`, `pollster 0.4`, `memmap2 0.9`, `serde`/`serde_json 1`,
`rand 0.9`, `wgpu 29` (`default-features = false`, `std` + `naga-ir`, backend
feature per target), `naga 29`, `fearless_simd 0.4`.

`naga` resolves to the same `29.0.4` wgpu itself depends on, so
`create_shader_module_trusted(naga::Module)` type-checks.

---

## 2. `fusor2-ir` — the contracts

Fifteen modules, in `lib.rs` declaration order:

```
error  dtype  shape  facts  scalar  ir{,::logical,::launch,::kernel}
egraph  device  cost  extract  target  autograd
carrier  contract_spec  semantics{,::children,::infer_l0,::infer_l1,::work}
verify_l0  verify_l1  rule_macro  rules{,::*}  saturate
```

### 2.1 Types you will use constantly

| Type | Module | Note |
|---|---|---|
| `Error`, `Result<T, E = Error>` | `error` | flat, `Clone`, `PartialEq` |
| `Dtype` = `F32\|F16\|BF16\|U32\|I32\|Q(QFmt)` | `dtype` | no `Bool`; comparisons are 1.0/0.0 |
| `QFmt` (6), `QLayout` | `dtype` | `BlockSpec`, `BlockProgram`, `GgmlType` live in `fusor2-gguf` |
| `NumericContract { min_accum_bits, reassoc, contract }` | `dtype` | `RELAXED`, `STRICT`, `allows`, `meet` |
| `Persistence`, `RoundMode`, `Splat` | `dtype` | `Splat` eq/hash are **bitwise** |
| `SymId`, `Dim`, `Dims`, `StrideSpec`, `SlidingWindow`, `Layout`, `MultiFlattenMap`, `BoundsProof` | `shape` | plus `broadcast_specs`, `broadcast_shapes` |
| `ValueFacts`, `Work` | `facts` | |
| `UnOp`(21), `BinOp`(15), `CmpOp`(6), `ScalarExpr`, `ScalarKind`, `Args` | `scalar` | `ScalarExpr::compose` **is** elementwise fusion |
| `Level`, `Op`, `Node`, `Children`, `OpTag`, `Semantics`, `VerifyCtx`, `OpDef`, `OpDefRegistry` | `ir` | |
| `Logical` (10 nodes), `LeafKind`, `TiePolicy`, `EinSpec` | `ir::logical` | |
| `Carrier`, `SlotTy`, `Tupled`, `ArgRemap`, `HOM_TABLE`, `RETARGET_TABLE` | `carrier` | the fold algebra as data; **no named combine enum** |
| `Launch` (10 variants), `IndexSpace`, `Operand`, `AccessPlan`, `ScheduleDomain`, `SchedPoint`, `Effect` | `ir::launch` | |
| the whole tile dialect + `ArenaPlanner` + `KernelIr` | `ir::kernel` | |
| `Id`, `ClassId`, `EGraph`, `Builder`, `Facts`, `Rule`, `RuleTag`, `Saturate` | `egraph` | |
| `Caps`, `Limits`, `CoopKind`, `DeviceKind` | `device` | `Limits::default()` is the **WebGPU baseline** |
| `Picoseconds`, `DeviceFacts`, `LaunchPlan`, `CostModel`, `ShapeStats` | `cost` | |
| `Extraction`, `Move`, `ExtractBudget`, `PlanHash`, `Plan`, `Extractor` | `extract` | |
| `Artifact`, `Buf`, `Uniforms`, `EmitError`, `LowerCtx`, `Target` | `target` | |
| `Val`, `Grads`, `Tape`, `AdjointFn`, `AdjointKind`, `Adjoint`, `Autograd` | `autograd` | |

### 2.2 The traits, verbatim

```rust
// ir::Semantics — object-safe; the e-graph holds Arc<dyn Semantics>
fn children(&self, op: &Op) -> Children;
fn infer(&self, op: &Op, ins: &[ValueFacts]) -> Result<ValueFacts>;
fn work(&self, op: &Op, ins: &[ValueFacts], out: &ValueFacts) -> Work;
fn verify(&self, cx: &VerifyCtx<'_>) -> Result<()>;
fn effect(&self, op: &Op) -> Effect;

// ir::kernel::ArenaPlanner — object-safe; one impl in fusor2-tile
fn arena_plan(&self, ir: &KernelIr, caps: &Caps) -> Result<ArenaPlan>;
fn workgroup_bytes(&self, tiles: &Tiles, caps: &Caps) -> Result<u32>;
fn barrier_suggestions(&self, ir: &KernelIr) -> Vec<BarrierSuggestion>;
fn verify_arena(&self, ir: &KernelIr, plan: &ArenaPlan) -> Result<()>;
fn verify_uniformity(&self, ir: &KernelIr) -> Result<()>;

// egraph::Saturate
fn saturate(&self, graph: &mut EGraph, caps: &Caps, rules: &[Rule],
            budget: SaturationBudget) -> Result<SaturationReport>;

// cost::CostModel — object-safe
fn facts(&self) -> &DeviceFacts;
fn launch_cost(&self, launch: &LaunchPlan<'_>) -> Picoseconds;
fn node_math(&self, node: &Node, ins: &[ValueFacts], out: &ValueFacts,
             theta: Option<SchedPoint>) -> Picoseconds;
fn traffic(&self, bytes: u64, rereads: u32) -> Picoseconds;
fn compile_amortized(&self, plan: PlanHash, expected_reuse: u32) -> Picoseconds;
fn total(&self, extraction: &Extraction, launches: &[LaunchPlan<'_>]) -> Picoseconds;

// extract::Extractor — object-safe
fn lower_bound(&self, graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds>;
fn extract(&self, graph: &EGraph, roots: &[Id], cost: &dyn CostModel,
           budget: ExtractBudget) -> Result<Plan>;
fn verify_plan(&self, graph: &EGraph, plan: &Plan) -> Result<()>;

// target::Target — object-safe; the session holds Arc<dyn Target>
fn name(&self) -> &'static str;
fn caps(&self) -> &Caps;
fn facts(&self) -> &DeviceFacts;
fn rules(&self) -> &'static [Rule];
fn lower(&self, node: &Node, id: Id, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr>;
fn emit(&self, ir: &KernelIr) -> std::result::Result<Artifact, EmitError>;
fn launch(&self, artifact: &Artifact, grid: [u32; 3], binds: &[Buf],
          uniforms: &Uniforms) -> Result<()>;
fn alloc(&self, bytes: u64, persistence: Persistence) -> Result<Buf>;
fn wait(&self) -> Result<()>;

// autograd::Tape — object-safe; AdjointFn takes &mut dyn Tape
fn add(&mut self, op: Logical) -> Result<Val>;
fn facts(&self, v: Val) -> &ValueFacts;
fn zeros_like(&mut self, v: Val) -> Result<Val>;
fn map(&mut self, expr: ScalarExpr, ins: &[Val]) -> Result<Val>;
fn contract(&mut self, a: Val, b: Val, spec: EinSpec, acc: Dtype) -> Result<Val>;
fn fold(&mut self, carrier: Carrier, axis: u32, acc: Dtype, x: Val) -> Result<Val>;
fn fold_binop(&mut self, op: BinOp, axis: u32, acc: Dtype, x: Val) -> Result<Val>;  // provided
fn restride(&mut self, specs: &[StrideSpec], x: Val) -> Result<Val>;
fn scatter_add(&mut self, axis: u32, base: Val, idx: Val, upd: Val) -> Result<Val>;
fn accumulate(&mut self, a: Val, b: Val) -> Result<Val>;

// autograd::Autograd
fn adjoints(&self) -> &'static [Adjoint];
fn backward(&self, tape: &mut dyn Tape, root: Val, seed: Val, wrt: &[Val])
    -> Result<Vec<Option<Val>>>;
```

### 2.3 The rule signature — read this before writing a guard

```rust
pub type RuleFn = fn(&mut Builder<'_>, Id, &Node, &Facts<'_>) -> Option<Id>;

pub struct Rule {
    pub name: &'static str,
    pub level: Level,
    pub head: OpTag,      // O(1) dispatch filter
    pub tag: RuleTag,     // Additive | StrictlyLowering
    pub apply: RuleFn,
}
```

`Facts<'a>` exposes `caps()`, `level()`, `own()`, `operand(slot)`, `operands()`,
`numeric(slot)`, `dim(slot, axis)`, `dtype(slot)`. It borrows **only** `Caps`,
never the graph — that is what lets the driver hand you a `&mut Builder` over the
same graph in the same call.

It exposes **no consumer counts, no liveness, no cost, no extraction state**, and
that is deliberate and load-bearing. **Guards encode legality only.** If you find
yourself wanting to know how many consumers a value has, you are writing a
profitability judgement; put it in `fusor2-cost` instead. Returning `None`
because something *would not pay* is a bug in this design, not a shortcut.

`Builder` gives you `caps`, `node`, `facts_of`, `level_of`, `add_l0`, `add_l1`,
`add`, `union`, `fresh_sym`, `mark_defn`, `trace_pure_views`, `spine_specs`.

### 2.4 The `rule!` macro (scaffold form)

```rust
rule!(
    FOLD_SPLIT,
    level = Level::Logical,
    head  = OpTag::Fold,
    tag   = RuleTag::Additive,
    apply = fold_split,
);
```

expands to `pub const FOLD_SPLIT: Rule = Rule { name: "FOLD_SPLIT", .. }`.

Every rule module already uses this form. **W2 owns `rule_macro.rs` and may add
structural-pattern arms**, but must keep this arm working, because eleven other
files call it.

### 2.5 E-graph invariants you must not break

- `children` may only hold ids **strictly smaller** than the node's own.
  `EGraph::add` enforces it and returns an error otherwise. Never construct a
  `Node` by hand.
- `union(a, b)` roots both chains first, then allocates a `Union` node above
  both, so a class stays complete under repeated unions.
- Equality is **not congruent**. Unioning `a` and `b` does not union `f(a)` and
  `f(b)`. Mint alternatives at the **consumer**, and use
  `Builder::trace_pure_views` when you need to match through a view spine.
- Macro ops union the sugar node and its `defn` expansion **in the same call**.
  There is no recognizer, and `mark_defn` nodes are never evicted.

---

## 3. Per-crate entry points

### `fusor2-tile` (W3, W4)

```rust
pub struct Planner;                       // impl ArenaPlanner, memoized
impl Planner { pub fn new() -> Self; pub fn shared() -> Arc<dyn ArenaPlanner>; }

pub fn verify_l2(ir: &KernelIr) -> Result<()>;
pub struct TileBuilder;                   // hash-consing Kernel term builders

pub mod liveness  { pub struct LiveRange; pub struct LivenessInfo;
                    pub fn analyze(&KernelIr) -> LivenessInfo; }
pub mod arena     { pub fn pack(..); regions(..); byte_arena(..); workgroup_bytes(..); }
pub mod barrier   { pub fn suggestions(..); insert(..); elide(..); }
pub mod uniformity{ pub enum Uniformity; pub fn verify_uniformity(..); expr_uniformity(..); }

pub mod domains {
    pub mod coop  { pub fn legal(m, n, k, operand, acc, &Caps) -> CoopDomain; }
    pub mod sgemm { pub fn legal(m, n, k, dtype, &Caps) -> SgemmDomain; }
    pub mod sgemv { pub fn legal(m, n, k, dtype, &Caps) -> SgemvDomain; }
    pub mod fold  { pub fn legal(axis_extent, rows, &Caps) -> FoldDomain; }
    pub mod map   { pub fn legal(&IndexSpace, &Caps) -> MapDomain; }
}

pub static SCHED_RULES: &[Rule];          // 15 rules, see rules.rs
```

### `fusor2-autograd` (W5)

```rust
pub struct Reverse;                       // impl Autograd
pub struct GraphTape<'a>;                 // impl Tape over &mut EGraph
pub static ADJOINTS: &[Adjoint];          // exactly 7 rows
pub fn map_adjoint(&mut dyn Tape, &Node, Val, &[Val], Val) -> Result<Grads>;
pub static ADJOINT_RULES: &[Rule];        // 3 recovery rules
```

`structural.rs` holds `restride_adjoint`, `window_adjoint`, `gather_adjoint`,
`scatter_adjoint`, `fold_adjoint`; `contract.rs` holds `contract_adjoint`. All
share `AdjointFn`'s shape even where `ADJOINTS` marks them `Structural`.

### `fusor2-cost` (W6, W7)

```rust
pub struct Roofline;                      // impl CostModel; Roofline::new(DeviceFacts)
pub struct LocalSearch;                   // impl Extractor
pub struct ReplayMemo;                    // get / insert, keyed on ReplayKey

pub mod facts       { pub fn seed_facts(&Caps) -> DeviceFacts; generic_seed(..); }
pub mod terms       { dram_ps, math_ps, wg_ps, drain_ps, occupancy_scale, swizzle_ps }
pub mod lower_bound { pub fn lower_bound(&EGraph, &dyn CostModel) -> Vec<Picoseconds>; }
pub mod realize     { pub struct Realized; realize(..); launches_of(..); forced_boundary(..); }
pub mod moves       { pub fn frontier(..); apply(..) -> Option<Undo>; undo(..);
                      is_pinned(..); evaluate(..); pub enum Undo; }
pub mod plan        { plan_hash, derive_buffers, derive_bindings, buffer_layout, symbols_of }
pub mod verify_plan { pub fn verify_plan(&EGraph, &Plan) -> Result<()>; }
```

### `fusor2-gguf` (W11)

```rust
pub static BLOCK_SPECS: &[BlockSpec];
pub fn block_spec(QFmt, QLayout) -> &'static BlockSpec;
pub fn repack(QFmt, from: QLayout, to: QLayout, &[u8]) -> Result<Vec<u8>>;
pub struct Gguf;  pub struct GgufMetadata;  pub struct GgufTensor;  pub enum GgufValue;
pub struct VarBuilder;  pub struct ShardedVarBuilder;  pub struct AsyncShardedVarBuilder;
pub trait AsyncReadRange { fn read_range(&self, start: u64, len: usize) -> ReadFuture<'_>; }
```

`decode.rs` holds the six 32-element programs, `decode_k.rs` the six K-quant
programs; each is a `BlockEmitFn = fn(&BlockDecodeArgs<'_>) -> Result<TileExpr>`.

### `fusor2-gpu` (W8, W9)

```rust
pub struct GpuTarget;    // impl Target; ::new().await, ::new_blocking()
pub struct GpuDevice;    // ::request(Option<wgpu::Limits>).await; caps/facts/limits_used
pub fn emit(&KernelIr, &Caps) -> Result<naga::Module, EmitError>;
pub struct Emitter<'a>;  // expr/stmt/reduce/coop/quantized hang off this as inherent impls
pub struct BindingDesc;  pub fn bindings_from_module(&naga::Module) -> Vec<BindingDesc>;
pub struct BufferPool;   // alloc / recycle / set_ceiling
pub struct PlanCache;    // get / insert / disk_salt
pub struct Launcher;     // encode / submit / poll_wait / take_kernel_profiles
pub static GPU_RULES: &[Rule];
pub mod lower { pub fn lower(&Caps, &Node, Id, SchedPoint, &LowerCtx) -> Result<KernelIr>; }
```

Each `lower::<family>` submodule exposes
`pub fn lower(&Caps, &Node, SchedPoint, &LowerCtx) -> Result<KernelIr>` (no `Id` —
the dispatcher in `lower.rs` has it).

Feature `fork-metal` gates the two extra capabilities; the default build must
work without it.

### `fusor2-cpu` (W10)

```rust
pub struct CpuTarget;    // impl Target; ::new()
pub struct CpuCaps;      // ::detect() -> Caps, ::llc_bytes(), ::threads()
pub struct AlignedBuf;   // 64-byte aligned
pub struct WorkerPool;   // ::global(), parallel_for(range, grain, &dyn Fn), num_threads()
pub struct CpuKernel;    // ::run(grid, binds, uniforms)
pub fn emit(&KernelIr, &Caps) -> Result<CpuKernel, EmitError>;
pub static CPU_RULES: &[Rule];   // 7 rules
pub mod emit::access { pub enum AccessForm { Contiguous, Broadcast, UnitInnerStride, Gather } }
pub mod emit::expr   { pub struct LaneValue { slot, width } }
pub mod emit::stmt   { pub struct LaneLoop; pub fn block(..) -> Result<Vec<LaneLoop>, _> }
```

`emit::stmt::block` returning `Vec<LaneLoop>` is the barrier split. It is not
optional: mapping `Barrier` to a no-op miscompiles every workgroup-staged kernel.

### `fusor2` (W12, W13)

```rust
pub struct Tensor { id: Id, graph: GraphRef }   // Clone; runtime rank + dtype
pub struct Typed<const R: usize>(Tensor);       // zero-cost, no IR effect
pub type GraphRef = Arc<GraphInner>;
pub struct GraphInner { egraph: Mutex<EGraph>, session: Session, params: Mutex<..> }
pub struct Graph;   // new/param/leaf/constant/sym/backward/backward_with/gradients
pub struct Gradients;                            // get(&Tensor) -> Option<Tensor>
pub struct Session; pub enum Device;             // resolve/flush/wait/launch_count
```

Op methods are inherent `impl Tensor` blocks spread across `ops/*.rs` and
`composite/*.rs`. Adding a method to *your* file needs no coordination; adding a
field to `Tensor` or `GraphInner` does — report it.

`Session::launch_count()` is what `resolves_in::<N>` reads. Keep it exact.

### `fusor2-conformance` (W14)

```rust
pub struct Harness;  pub struct Case;  pub enum Outcome;
pub static REGISTRY: &[Case];        // suite::<area>::CASES concatenated
pub fn allclose(&[f32], &[f32], atol, rtol) -> bool;
pub fn assert_close(..) -> Result<(), String>;
pub fn resolves_in<const N: u64>(&Session, &[Tensor]) -> Result<(), String>;
pub const NAMED_BACKWARD_SHAPES: [&str; 8];
pub struct IlpExtractor;             // impl Extractor, debug oracle, MAX_NODES = 64
pub mod goldens { pub struct Golden; plan_hash_goldens(); check(name, PlanHash); }
// `trainer_gate` (THROUGHPUT_FLOOR / PARITY_FLOOR / MSQ1_BYTES) and `msq1`
// (int4_quantize, ternary_quantize, pack, unpack) are NOT in this checkout:
// both were built on betlang's `trainer/src/*.rs` and its shipped
// `assets/magika/` artifact, neither of which lives in this repository.
```

---

## 4. Changes the scaffold made to the stated contracts

Every deviation from the architecture document's prose, and why. All are
surgical; none changes semantics.

1. **`domains::{CoopDomain::legal, SgemmDomain::legal, ...}` became free
   functions.** `CoopDomain`, `SgemmDomain`, `SgemvDomain`, `FoldDomain` and
   `MapDomain` are defined in `fusor2-ir`, so `fusor2-tile` cannot add inherent
   methods to them (orphan rule). They are now
   `fusor2_tile::domains::coop::legal(..)` etc., re-exported from `domains` as
   `coop_legal`, `sgemm_legal`, `sgemv_legal`, `fold_legal`, `map_legal`. Return
   types are unchanged.

2. **`verify_l1` takes the planner explicitly:**
   `verify_l1(cx: &VerifyCtx<'_>, planner: &dyn ArenaPlanner) -> Result<()>`.
   Invariant 2 admits a geometry against the *exact* `arena_plan` value, so the
   verifier needs the planner. `CoreSemantics` holds one and passes it through
   `Semantics::verify`, whose signature is unchanged. `verify_l0` is the stated
   `fn(&VerifyCtx<'_>) -> Result<()>`.

3. **`CostModel::emit`-adjacent `Target::emit` returns
   `std::result::Result<Artifact, EmitError>`**, not the crate `Result` alias —
   as the contract listing specifies. Note the explicit path in impls, because
   `crate::Result` is in scope in most files.

4. **On-disk formats live in `fusor2-cost` as mirror structs, never as serde
   derives on `fusor2-ir` types.** `fusor2-ir` has no `serde` dependency at all,
   and must not gain one: deriving on `DeviceFacts` or `Caps` would put a
   serialization format in the contracts crate and would persist a capability
   set, letting a stale one outlive a driver update. `fusor2-cost::tune_cache`
   is the live instance of the pattern — its `Record`/`Disk` mirrors round-trip
   the verdicts while `Caps` is always re-probed.

5. **The `rule!` macro ships only the declarative arm** described in §2.4. The
   architecture's structural-pattern form is W2's to add; the declarative arm is
   already used by all eighteen `fusor2-ir` rules, fifteen `fusor2-tile` rules,
   three `fusor2-autograd` rules and seven `fusor2-cpu` rules, so it must keep
   working.

6. **Rule constants are `SCREAMING_CASE`** (`FOLD_SPLIT`, `TILE_FOLD`, …)
   because `rule!` declares a `const`; the *function* keeps the architecture's
   lowercase name (`fold_split`, `tile_fold`). `Rule::name` is the constant's
   identifier, so conformance asserts fire on `"FOLD_SPLIT"`.

7. **`Extraction::sigma` is `FxHashMap<ClassId, Id>`.** The architecture's §4.1
   sketch writes `FxHashMap<Id, Id>` with the comment "union-chain root ->
   selected node"; the contracts source uses the newtype, which is the same thing
   with the intent in the type. `ClassId(Id)` is a transparent newtype.

8. **`fusor2-conformance` gained `src/main.rs`.** The crate is described as
   "Binary + test harness" but no binary file was listed; the scaffold owns it.
   It parses `--exhaustive` and will dispatch into `harness::run`.

9. **`CoreSemantics` gained `with_registry(Arc<dyn ArenaPlanner>, OpDefRegistry)`**
   alongside the stated `new(Arc<dyn ArenaPlanner>) -> Arc<dyn Semantics>`,
   because `Launch::Ext` needs a populated registry and `new` gives an empty one.

10. **Each `fusor2-gpu::lower::<family>` / `fusor2-cpu::lower::<family>`
    submodule's `lower` omits the `Id` parameter** that the top-level
    `lower::lower` takes. The dispatcher already resolved it; threading it further
    bought nothing.

Everything else — every type, field, method name, trait method signature and
constant in the contracts listing — is verbatim. In particular `Logical`'s ten nodes,
`Launch`'s ten variants, the full Kernel dialect, `Layout`'s private fields with derived
contiguity, `Splat`'s bitwise eq/hash, and
`Limits::default()`'s WebGPU baseline are exactly as specified.

---

## 5. Two things that will bite you

**`Facts` has no consumer count on purpose.** If a rule you are porting from the
reference had a `consumer_count(x) != 1` gate, a `skip_externally_live`, a
`variant_duplicates_required_producer`, or a `merge_profile -> None`, that gate
does not move to `fusor2-ir`. Delete it, mint the alternative unconditionally,
and let the realized-DAG cost in `fusor2-cost` reject it. Those four gates are
exactly the phase-ordering bugs this design exists to remove.

**`arena_plan` must be the same function at Launch and Kernel.** `verify_l1`'s footprint
check, the Launch occupancy term and the Kernel emitter's layout all read
`Planner::arena_plan`, memoized on `(geom, dtype, caps)`. If you add an estimator
anywhere, you have reintroduced "extraction commits a plan that fails Kernel
verification and silently falls back".
