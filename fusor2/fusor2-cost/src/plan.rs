//! Plan derivation: buffers, bindings, symbols and the plan hash.
//!
//! **Allocation is derived from the plan**: for each node in `M`,
//! [`buffer_layout`] gives the padded strides the selected geometry needs,
//! including split-K scratch slices. `hardware_matmul_prep`'s exact-stride
//! equality test and its silent generic-reduce fallback become an invariant
//! the extractor establishes rather than a runtime test. A value not in `M`
//! gets no buffer at all, which subsumes the reference's `BufferLedger`.
//!
//! **The plan is the cache key.** `Dim::Sym` and `LeafKind::Uniform` hash as
//! the symbol's *index*, not its bound value, so one plan serves a whole
//! shape family. There is no `hash_kernel_fields`, no
//! `kernel_cache_key_with_dispatch`, no `structural_kernel_key` and no golden
//! byte files.
//!
//! Owned by W7.

use crate::realize::{self, Component, Realized};
use fusor2_ir::Result;
use fusor2_ir::cost::DeviceFacts;
use fusor2_ir::egraph::{EGraph, Id};
use fusor2_ir::extract::{BindKind, BindingPlan, BufferPlan, Extraction, Launch, Plan, PlanHash};
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::ir::Op;
use fusor2_ir::ir::level0::{L0, LeafKind};
use fusor2_ir::ir::level1::{AccessPlan, Effect, IndexSpace, L1, Operand, SchedPoint};
use fusor2_ir::scalar::{ScalarExpr, ScalarKind};
use fusor2_ir::shape::{Dim, Dims, Layout, SymId};
use rustc_hash::FxHasher;
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// `SymId(u32::MAX)` is the crate-wide "symbolic, not statically known"
/// sentinel — `Layout::row_major_strides` already mints it for a stride past
/// a symbolic axis. A `BufferPlan::elements` of this value means the runtime
/// derives the extent from `layout` plus the bound symbols.
pub const UNKNOWN_SYM: SymId = SymId(u32::MAX);

/// Everything derived from one realized extraction: buffers, launches,
/// symbols, hash and cost.
pub fn derive_plan(
    graph: &EGraph,
    extraction: &Extraction,
    realized: &Realized,
    facts: &DeviceFacts,
    cost: fusor2_ir::cost::Picoseconds,
) -> Result<Plan> {
    let buffers = derive_buffers(graph, extraction, realized)?;
    let symbols = symbols_of(graph, realized);

    let mut launches = Vec::with_capacity(realized.components.len());
    for c in &realized.components {
        launches.push(Launch {
            root: c.root,
            members: c.members.iter().copied().collect(),
            bindings: derive_bindings(graph, extraction, realized, c)?,
            grid: c.grid,
            block: c.block,
        });
    }

    let hash = plan_hash(graph, extraction, &launches, &symbols, facts);
    Ok(Plan {
        extraction: extraction.clone(),
        launches,
        buffers,
        symbols,
        hash,
        cost,
    })
}

/// One [`BufferPlan`] per node in `m ∪ roots`, in realized order. Leaves are
/// excluded: an external buffer is supplied, a constant is folded, and a
/// uniform lives in binding 0.
pub fn derive_buffers(
    graph: &EGraph,
    extraction: &Extraction,
    realized: &Realized,
) -> Result<Vec<BufferPlan>> {
    let mut out = Vec::new();
    for id in &realized.order {
        if realize::leaf_role(graph, *id) != realize::LeafRole::NotLeaf {
            continue;
        }
        if !extraction.is_materialized(*id) && !realized.is_root(*id) {
            continue;
        }
        let facts = graph.facts(*id);
        let theta = extraction.theta.get(id).copied();
        let layout = buffer_layout_for(facts, theta)?;
        out.push(BufferPlan {
            value: *id,
            elements: layout_elements(&layout),
            layout,
            dtype: facts.dtype,
            persistence: facts.persistence,
        });
    }
    Ok(out)
}

/// Bindings of one launch, in binding-index order. **Binding 0 is reserved
/// for the uniform block** and is never listed here; storage bindings start
/// at 1, reads first sorted by value id, then writes.
pub fn derive_bindings(
    graph: &EGraph,
    extraction: &Extraction,
    realized: &Realized,
    component: &Component,
) -> Result<Vec<BindingPlan>> {
    let mut writes: Vec<Id> = component
        .members
        .iter()
        .copied()
        .filter(|m| extraction.is_materialized(*m) || realized.is_root(*m))
        .collect();
    writes.sort_unstable();
    writes.dedup();

    let mut reads: Vec<Id> = component.external.clone();
    reads.retain(|r| realize::leaf_role(graph, *r) != realize::LeafRole::Free);
    reads.sort_unstable();
    reads.dedup();
    // An in-place value is bound once, read-write; it must not appear twice.
    reads.retain(|r| !writes.contains(r));

    let mut out = Vec::with_capacity(reads.len() + writes.len());
    let mut binding = 1u32;
    for value in reads {
        out.push(BindingPlan {
            binding,
            value,
            kind: BindKind::Read,
        });
        binding += 1;
    }
    for value in writes {
        let kind = match graph.semantics().effect(&graph.node(value).op) {
            Effect::InPlace(_) => BindKind::ReadWrite,
            Effect::Pure => BindKind::Write,
        };
        out.push(BindingPlan {
            binding,
            value,
            kind,
        });
        binding += 1;
    }
    Ok(out)
}

/// The padded layout one materialized value needs under one schedule point.
///
/// - default: `Layout::contiguous(shape)`;
/// - `Coop { geom, splits, .. }`: pad `m` to a multiple of `geom.bm` and `n`
///   to `geom.bn`, row-major strides over the padded shape, and when
///   `splits > 1` prepend a leading axis of extent `splits` with stride
///   `padded_m * padded_n` — the split-K scratch slice;
/// - `Sgemm` / `Sgemv` / `Fold` / `Map` / `Point`: contiguous.
///
/// **`Sgemm` pads nothing.** Padding exists so a kernel may write a whole
/// block without a bounds test, and only the cooperative store does that: the
/// SGEMM body masks every store with `row < batch * m && col < n`. Padding it
/// anyway was not merely wasteful, it was unsound — this function pads the
/// output's last *two axes*, which are the `m` and `n` axes only when each
/// occupies exactly one, and a contraction with `n = 1` (every row reduction
/// of a product: rms_norm, layer_norm, variance) has none. Its `[rows, cols]`
/// output was being padded along two batch axes instead.
pub fn buffer_layout_for(facts: &ValueFacts, theta: Option<SchedPoint>) -> Result<Layout> {
    let shape = &facts.shape;
    let (bm, bn, splits) = match theta {
        Some(SchedPoint::Coop { geom, splits, .. }) => (geom.bm, geom.bn, splits),
        _ => return Ok(Layout::contiguous(shape)),
    };
    if shape.len() < 2 {
        return Ok(Layout::contiguous(shape));
    }

    let mut padded: Dims = shape.clone();
    let last = padded.len() - 1;
    padded[last - 1] = pad_to(padded[last - 1], bm);
    padded[last] = pad_to(padded[last], bn);

    if splits <= 1 {
        return Ok(Layout::contiguous(&padded));
    }

    // Split-K scratch: one whole padded output per partial, so the combine
    // pass reads slice `s` one *whole output* in.
    //
    // That distance is the product of every padded extent, batch axes
    // included — it is exactly the row-major stride a prepended axis gets,
    // `strides[0] * padded[0]`. It used to be `padded_m * padded_n`, which is
    // one batch element rather than one partial: with any leading batch axis,
    // partial `s` began inside partial `s-1` and every batch past the first
    // aliased.
    let strides = Layout::row_major_strides(&padded);
    let slice = match (
        strides.first().and_then(|s| s.as_const()),
        padded.first().and_then(|d| d.as_const()),
    ) {
        (Some(outer_stride), Some(outer_extent)) => Dim::Const(outer_stride * outer_extent),
        _ => Dim::Sym(UNKNOWN_SYM),
    };
    let mut shape_out: Dims = smallvec::smallvec![Dim::Const(splits as u64)];
    shape_out.extend(padded.iter().copied());
    let mut strides_out: SmallVec<[Dim; 6]> = smallvec::smallvec![slice];
    strides_out.extend(strides.iter().copied());
    Layout::from_parts(Dim::Const(0), &shape_out, &strides_out)
}

const fn pad_to(d: Dim, multiple: u32) -> Dim {
    match (d.as_const(), multiple) {
        (Some(v), m) if m > 1 => Dim::Const(v.div_ceil(m as u64) * m as u64),
        _ => d,
    }
}

fn layout_elements(layout: &Layout) -> Dim {
    let mut acc: u64 = 1;
    for d in layout.shape() {
        match d.as_const() {
            Some(v) => acc = acc.saturating_mul(v),
            None => return Dim::Sym(UNKNOWN_SYM),
        }
    }
    Dim::Const(acc)
}

/// Every `SymId` the uniform block must carry, in binding order: dims
/// ascending, then scalars ascending, matching `Uniforms::to_bytes`.
pub fn symbols_of(graph: &EGraph, realized: &Realized) -> Vec<SymId> {
    let mut dims: Vec<SymId> = Vec::new();
    let mut scalars: Vec<SymId> = Vec::new();

    for id in &realized.order {
        collect_dims(&graph.facts(*id).shape, &mut dims);
        let op = &graph.node(*id).op;
        collect_op(op, &mut dims, &mut scalars);
    }

    dims.retain(|s| *s != UNKNOWN_SYM);
    scalars.retain(|s| *s != UNKNOWN_SYM);
    dims.sort_unstable();
    dims.dedup();
    scalars.sort_unstable();
    scalars.dedup();
    // A symbol used as an extent is bound as a dim; it must not also be
    // emitted as a scalar.
    scalars.retain(|s| !dims.contains(s));

    dims.extend(scalars);
    dims
}

/// `hash(realized DAG term + M + theta + DeviceFacts::fingerprint)`.
///
/// Two `FxHasher` lanes under seeds 0 and 1, folded into a `u128`. Walk
/// launches in order, then members in order; `Dim::Sym(s)` and
/// `LeafKind::Uniform { sym }` hash as the symbol's index in `symbols`,
/// never its bound value.
pub fn plan_hash(
    graph: &EGraph,
    extraction: &Extraction,
    launches: &[Launch],
    symbols: &[SymId],
    facts: &DeviceFacts,
) -> PlanHash {
    let mut lanes = [FxHasher::default(), FxHasher::default()];
    let sm = SymMap::new(symbols);
    for (seed, h) in lanes.iter_mut().enumerate() {
        h.write_u64(seed as u64);
        for launch in launches {
            h.write_u32(launch.root.0);
            h.write_u32(launch.grid[0]);
            h.write_u32(launch.grid[1]);
            h.write_u32(launch.grid[2]);
            h.write_u32(launch.block);
            for b in &launch.bindings {
                h.write_u32(b.binding);
                h.write_u32(b.value.0);
                (b.kind as u8).hash(h);
            }
            for member in &launch.members {
                h.write_u32(member.0);
                hash_op(h, &sm, &graph.node(*member).op);
                // Leaf operands are never launch members, so their kind
                // would otherwise never reach the hash. Their *name* stays
                // out: buffer identity is deliberately absent from the key,
                // which is what lets a bufferless template rebind
                // positionally.
                for child in graph.node(*member).children.iter() {
                    if let Op::L0(L0::Leaf(kind)) = &graph.node(*child).op {
                        hash_leaf_ref(h, &sm, kind);
                    }
                }
                h.write_u8(u8::from(extraction.is_materialized(*member)));
                match extraction.theta.get(member) {
                    Some(t) => {
                        h.write_u8(1);
                        t.hash(h);
                    }
                    None => h.write_u8(0),
                }
            }
        }
        h.write_u64(facts.fingerprint());
    }
    PlanHash(((lanes[0].finish() as u128) << 64) | lanes[1].finish() as u128)
}

// ---------------------------------------------------------------------------
// Symbol-aware structural hashing
// ---------------------------------------------------------------------------

struct SymMap<'a> {
    symbols: &'a [SymId],
    /// Symbol-remapped digest per `ScalarExpr::structural_hash`. Repeated
    /// transformer layers and a 3,000-node conv step share a handful of
    /// distinct bodies, so this collapses the walk to one per body.
    memo: std::cell::RefCell<rustc_hash::FxHashMap<u64, u64>>,
}

impl<'a> SymMap<'a> {
    fn new(symbols: &'a [SymId]) -> Self {
        Self {
            symbols,
            memo: std::cell::RefCell::new(rustc_hash::FxHashMap::default()),
        }
    }

    /// The symbol's *index*, so two bindings of the same family collide and
    /// two structurally different plans do not.
    fn idx(&self, s: SymId) -> u32 {
        self.symbols
            .iter()
            .position(|x| *x == s)
            .map_or(u32::MAX, |i| i as u32)
    }

    fn scalar_digest(&self, e: &ScalarExpr) -> u64 {
        let key = e.structural_hash();
        if let Some(hit) = self.memo.borrow().get(&key) {
            return *hit;
        }
        let mut h = FxHasher::default();
        hash_scalar_uncached(&mut h, self, e);
        let v = h.finish();
        self.memo.borrow_mut().insert(key, v);
        v
    }
}

/// A leaf as an *operand*: everything that changes the kernel body, and
/// nothing that only names a buffer.
fn hash_leaf_ref<H: Hasher>(h: &mut H, sm: &SymMap<'_>, kind: &LeafKind) {
    std::mem::discriminant(kind).hash(h);
    match kind {
        LeafKind::Buffer { dtype, shape, .. } | LeafKind::Param { dtype, shape, .. } => {
            dtype.hash(h);
            hash_dims(h, sm, shape);
        }
        LeafKind::Const { value, shape } => {
            value.hash(h);
            hash_dims(h, sm, shape);
        }
        LeafKind::Uniform { sym, dtype } => {
            h.write_u32(sm.idx(*sym));
            dtype.hash(h);
        }
        LeafKind::Quantized {
            fmt, layout, shape, ..
        } => {
            fmt.hash(h);
            layout.hash(h);
            hash_dims(h, sm, shape);
        }
    }
}

fn hash_dim<H: Hasher>(h: &mut H, sm: &SymMap<'_>, d: Dim) {
    match d {
        Dim::Const(v) => {
            h.write_u8(0);
            h.write_u64(v);
        }
        Dim::Sym(s) => {
            h.write_u8(1);
            h.write_u32(sm.idx(s));
        }
    }
}

fn hash_dims<H: Hasher>(h: &mut H, sm: &SymMap<'_>, ds: &[Dim]) {
    h.write_usize(ds.len());
    for d in ds {
        hash_dim(h, sm, *d);
    }
}

fn hash_layout<H: Hasher>(h: &mut H, sm: &SymMap<'_>, l: &Layout) {
    hash_dim(h, sm, l.offset());
    hash_dims(h, sm, l.shape());
    hash_dims(h, sm, l.strides());
}

fn hash_space<H: Hasher>(h: &mut H, sm: &SymMap<'_>, s: &IndexSpace) {
    hash_dims(h, sm, &s.dims);
}

fn hash_scalar<H: Hasher>(h: &mut H, sm: &SymMap<'_>, e: &ScalarExpr) {
    h.write_u64(sm.scalar_digest(e));
}

fn hash_scalar_uncached<H: Hasher>(h: &mut H, sm: &SymMap<'_>, e: &ScalarExpr) {
    e.dtype().hash(h);
    match e.kind() {
        ScalarKind::Arg(i) => {
            h.write_u8(0);
            h.write_u32(*i);
        }
        ScalarKind::Lit(l) => {
            h.write_u8(1);
            l.hash(h);
        }
        ScalarKind::Uniform(s) => {
            h.write_u8(2);
            h.write_u32(sm.idx(*s));
        }
        ScalarKind::IndexOf(a) => {
            h.write_u8(3);
            h.write_u32(*a);
        }
        ScalarKind::Un { op, x } => {
            h.write_u8(4);
            op.hash(h);
            hash_scalar(h, sm, x);
        }
        ScalarKind::Bin { op, a, b } => {
            h.write_u8(5);
            op.hash(h);
            hash_scalar(h, sm, a);
            hash_scalar(h, sm, b);
        }
        ScalarKind::Cmp { op, a, b } => {
            h.write_u8(6);
            op.hash(h);
            hash_scalar(h, sm, a);
            hash_scalar(h, sm, b);
        }
        ScalarKind::Select { c, t, f } => {
            h.write_u8(7);
            hash_scalar(h, sm, c);
            hash_scalar(h, sm, t);
            hash_scalar(h, sm, f);
        }
        ScalarKind::Cast { to, x } => {
            h.write_u8(8);
            to.hash(h);
            hash_scalar(h, sm, x);
        }
        ScalarKind::Bitcast { to, x } => {
            h.write_u8(9);
            to.hash(h);
            hash_scalar(h, sm, x);
        }
        ScalarKind::Round { mode, x } => {
            h.write_u8(10);
            mode.hash(h);
            hash_scalar(h, sm, x);
        }
        ScalarKind::Dot { a, b } => {
            h.write_u8(11);
            hash_scalar(h, sm, a);
            hash_scalar(h, sm, b);
        }
        ScalarKind::Splat { lanes, x } => {
            h.write_u8(12);
            h.write_u32(*lanes);
            hash_scalar(h, sm, x);
        }
    }
}

fn hash_operand<H: Hasher>(h: &mut H, sm: &SymMap<'_>, o: &Operand) {
    h.write_u32(o.src.0);
    hash_layout(h, sm, &o.layout);
    match &o.access {
        AccessPlan::Alias => h.write_u8(0),
        AccessPlan::Gather => h.write_u8(1),
        AccessPlan::Pack { into } => {
            h.write_u8(2);
            hash_layout(h, sm, into);
        }
        AccessPlan::Unflatten(map) => {
            h.write_u8(3);
            map.hash(h);
        }
    }
}

fn hash_operands<H: Hasher>(h: &mut H, sm: &SymMap<'_>, ops: &[Operand]) {
    h.write_usize(ops.len());
    for o in ops {
        hash_operand(h, sm, o);
    }
}

fn hash_op<H: Hasher>(h: &mut H, sm: &SymMap<'_>, op: &Op) {
    op.tag().hash(h);
    match op {
        Op::Union(a, b) => {
            h.write_u32(a.0);
            h.write_u32(b.0);
        }
        Op::L0(l0) => hash_l0(h, sm, l0),
        Op::L1(l1) => hash_l1(h, sm, l1),
    }
}

fn hash_l0<H: Hasher>(h: &mut H, sm: &SymMap<'_>, op: &L0) {
    match op {
        L0::Leaf(k) => match k {
            LeafKind::Buffer { name, dtype, shape } | LeafKind::Param { name, dtype, shape } => {
                name.hash(h);
                dtype.hash(h);
                hash_dims(h, sm, shape);
            }
            LeafKind::Const { value, shape } => {
                value.hash(h);
                hash_dims(h, sm, shape);
            }
            // The bound value never appears in the IR and never enters the
            // hash: only the symbol's slot in the uniform block does.
            LeafKind::Uniform { sym, dtype } => {
                h.write_u32(sm.idx(*sym));
                dtype.hash(h);
            }
            LeafKind::Quantized {
                name,
                fmt,
                layout,
                shape,
            } => {
                name.hash(h);
                fmt.hash(h);
                layout.hash(h);
                hash_dims(h, sm, shape);
            }
        },
        L0::Map { expr, ins, outs } => {
            hash_scalar(h, sm, expr);
            for i in ins {
                h.write_u32(i.0);
            }
            h.write_u8(*outs);
        }
        L0::Fold {
            carrier,
            axis,
            acc,
            ins,
        } => {
            hash_carrier(h, sm, carrier);
            h.write_u32(*axis);
            acc.hash(h);
            for i in ins {
                h.write_u32(i.0);
            }
        }
        L0::Contract {
            spec,
            acc,
            a,
            b,
            outs,
        } => {
            spec.hash(h);
            acc.hash(h);
            h.write_u32(a.0);
            h.write_u32(b.0);
            h.write_u8(*outs);
        }
        L0::Restride { specs, bounds, x } => {
            h.write_usize(specs.len());
            for s in specs {
                h.write_u32(s.input_dim);
                h.write_u32(s.multiplier);
                hash_dim(h, sm, s.size);
                hash_dim(h, sm, s.offset);
            }
            bounds.hash(h);
            h.write_u32(x.0);
        }
        L0::Window { specs, x } => {
            specs.hash(h);
            h.write_u32(x.0);
        }
        L0::Gather { axis, x, idx } => {
            h.write_u32(*axis);
            h.write_u32(x.0);
            h.write_u32(idx.0);
        }
        L0::Scatter {
            axis,
            combine,
            base,
            idx,
            upd,
            unique,
        } => {
            h.write_u32(*axis);
            combine.hash(h);
            h.write_u32(base.0);
            h.write_u32(idx.0);
            h.write_u32(upd.0);
            h.write_u8(u8::from(*unique));
        }
        L0::Dequant { fmt, layout, x } => {
            fmt.hash(h);
            layout.hash(h);
            h.write_u32(x.0);
        }
        L0::Project { slot, x } => {
            h.write_u8(*slot);
            h.write_u32(x.0);
        }
    }
}

/// A carrier enters the plan hash as data: slot shapes, identities, and both
/// expression vectors. Two folds that differ only in their merge are different
/// kernels, and nothing about that is derivable from a name any more.
fn hash_carrier<H: Hasher>(h: &mut H, sm: &SymMap<'_>, c: &fusor2_ir::carrier::Carrier) {
    h.write_usize(c.slots.len());
    for s in &c.slots {
        match s {
            fusor2_ir::carrier::SlotTy::Scalar => h.write_u8(0),
            fusor2_ir::carrier::SlotTy::Vector(d) => {
                h.write_u8(1);
                hash_dim(h, sm, *d);
            }
        }
    }
    for i in &c.identity {
        i.hash(h);
    }
    for e in c.lift.iter().chain(&c.merge) {
        hash_scalar(h, sm, e);
    }
    c.associative.hash(h);
    c.tie.hash(h);
}

fn collect_carrier(
    c: &fusor2_ir::carrier::Carrier,
    dims: &mut Vec<SymId>,
    scalars: &mut Vec<SymId>,
) {
    for s in &c.slots {
        if let fusor2_ir::carrier::SlotTy::Vector(d) = s {
            collect_dims(&[*d], dims);
        }
    }
    for e in c.lift.iter().chain(&c.merge) {
        collect_scalar(e, scalars);
    }
}

fn hash_l1<H: Hasher>(h: &mut H, sm: &SymMap<'_>, op: &L1) {
    match op {
        L1::KMap {
            space, body, ops, ..
        } => {
            hash_space(h, sm, space);
            hash_scalar(h, sm, body);
            hash_operands(h, sm, ops);
        }
        L1::KFold {
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            ops,
            ..
        } => {
            hash_space(h, sm, space);
            h.write_u32(*axis);
            for a in vec_axes {
                h.write_u32(*a);
            }
            hash_carrier(h, sm, carrier);
            acc.hash(h);
            for p in post {
                hash_scalar(h, sm, p);
            }
            hash_operands(h, sm, ops);
        }
        L1::KContract {
            m,
            n,
            k,
            batch,
            family,
            post,
            acc,
            a,
            b,
            ..
        } => {
            hash_dim(h, sm, *m);
            hash_dim(h, sm, *n);
            hash_dim(h, sm, *k);
            hash_dim(h, sm, *batch);
            family.hash(h);
            hash_scalar(h, sm, &a.pre);
            hash_scalar(h, sm, &b.pre);
            hash_scalar(h, sm, post);
            acc.hash(h);
            // Arity first: a kernel keyed only on the operands it happens to
            // list would collide a two-buffer contraction with a wider one
            // whose extra edges hash the same way.
            h.write_usize(a.len());
            h.write_usize(b.len());
            for o in a.ops.iter().chain(b.ops.iter()) {
                hash_operand(h, sm, o);
            }
        }
        L1::KGather {
            space,
            axis,
            mode,
            ops,
            ..
        } => {
            hash_space(h, sm, space);
            h.write_u32(*axis);
            mode.hash(h);
            hash_operands(h, sm, ops);
        }
        L1::KScatter {
            space,
            axis,
            mode,
            combine,
            ops,
            ..
        } => {
            hash_space(h, sm, space);
            h.write_u32(*axis);
            mode.hash(h);
            combine.hash(h);
            hash_operands(h, sm, ops);
        }
        L1::KRegion {
            members, live_outs, ..
        } => {
            for m in members {
                h.write_u32(m.0);
            }
            live_outs.hash(h);
        }
        L1::Ext { def, ops, attrs } => {
            def.hash(h);
            hash_operands(h, sm, ops);
            attrs.hash(h);
        }
    }
}

// ---------------------------------------------------------------------------
// Symbol collection
// ---------------------------------------------------------------------------

fn collect_dims(dims: &[Dim], out: &mut Vec<SymId>) {
    for d in dims {
        if let Dim::Sym(s) = d {
            out.push(*s);
        }
    }
}

fn collect_layout(l: &Layout, out: &mut Vec<SymId>) {
    collect_dims(&[l.offset()], out);
    collect_dims(l.shape(), out);
    collect_dims(l.strides(), out);
}

fn collect_scalar(e: &ScalarExpr, scalars: &mut Vec<SymId>) {
    match e.kind() {
        ScalarKind::Uniform(s) => scalars.push(*s),
        ScalarKind::Arg(_) | ScalarKind::Lit(_) | ScalarKind::IndexOf(_) => {}
        ScalarKind::Un { x, .. }
        | ScalarKind::Cast { x, .. }
        | ScalarKind::Bitcast { x, .. }
        | ScalarKind::Round { x, .. }
        | ScalarKind::Splat { x, .. } => collect_scalar(x, scalars),
        ScalarKind::Bin { a, b, .. } | ScalarKind::Cmp { a, b, .. } | ScalarKind::Dot { a, b } => {
            collect_scalar(a, scalars);
            collect_scalar(b, scalars);
        }
        ScalarKind::Select { c, t, f } => {
            collect_scalar(c, scalars);
            collect_scalar(t, scalars);
            collect_scalar(f, scalars);
        }
    }
}

fn collect_ops(ops: &[Operand], dims: &mut Vec<SymId>) {
    for o in ops {
        collect_layout(&o.layout, dims);
        if let AccessPlan::Pack { into } = &o.access {
            collect_layout(into, dims);
        }
    }
}

fn collect_op(op: &Op, dims: &mut Vec<SymId>, scalars: &mut Vec<SymId>) {
    match op {
        Op::Union(..) => {}
        Op::L0(l0) => match l0 {
            L0::Leaf(LeafKind::Uniform { sym, .. }) => scalars.push(*sym),
            L0::Leaf(
                LeafKind::Buffer { shape, .. }
                | LeafKind::Param { shape, .. }
                | LeafKind::Const { shape, .. },
            ) => collect_dims(shape, dims),
            L0::Leaf(LeafKind::Quantized { shape, .. }) => collect_dims(shape, dims),
            L0::Map { expr, .. } => collect_scalar(expr, scalars),
            L0::Fold { carrier, .. } => {
                collect_carrier(carrier, dims, scalars);
            }
            L0::Restride { specs, .. } => {
                for s in specs {
                    collect_dims(&[s.size, s.offset], dims);
                }
            }
            _ => {}
        },
        Op::L1(l1) => match l1 {
            L1::KMap {
                space, body, ops, ..
            } => {
                collect_dims(&space.dims, dims);
                collect_scalar(body, scalars);
                collect_ops(ops, dims);
            }
            L1::KFold {
                space,
                carrier,
                post,
                ops,
                ..
            } => {
                collect_dims(&space.dims, dims);
                collect_carrier(carrier, dims, scalars);
                for p in post {
                    collect_scalar(p, scalars);
                }
                collect_ops(ops, dims);
            }
            L1::KContract {
                m,
                n,
                k,
                batch,
                post,
                a,
                b,
                ..
            } => {
                collect_dims(&[*m, *n, *k, *batch], dims);
                collect_scalar(&a.pre, scalars);
                collect_scalar(&b.pre, scalars);
                collect_scalar(post, scalars);
                collect_ops(&a.ops, dims);
                collect_ops(&b.ops, dims);
            }
            L1::KGather { space, ops, .. } | L1::KScatter { space, ops, .. } => {
                collect_dims(&space.dims, dims);
                collect_ops(ops, dims);
            }
            L1::Ext { ops, .. } => collect_ops(ops, dims),
            L1::KRegion { .. } => {}
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realize::testkit::{TestCost, TestPlanner, seed_for};
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::ir::level1::CoopGeom;

    fn f32_facts(shape: &[Dim]) -> ValueFacts {
        ValueFacts::new(Dtype::F32, shape.iter().copied())
    }

    fn geom(bm: u32, bn: u32) -> CoopGeom {
        CoopGeom {
            bm,
            bn,
            bk: 32,
            n_passes: 1,
            subgroups: 4,
            rg: 2,
            cg: 2,
        }
    }

    #[test]
    fn split_k_scratch_slice() {
        let facts = f32_facts(&[Dim::Const(512), Dim::Const(512)]);
        let theta = SchedPoint::Coop {
            geom: geom(64, 64),
            splits: 4,
            staging: 2,
        };
        let layout = buffer_layout_for(&facts, Some(theta)).unwrap();
        assert_eq!(layout.rank(), 3);
        assert_eq!(layout_elements(&layout), Dim::Const(4 * 512 * 512));
        assert_eq!(layout.strides()[0], Dim::Const(512 * 512));
        assert_eq!(layout.shape()[0], Dim::Const(4));
    }

    /// One partial is one *whole* output, batch axes included.
    ///
    /// The stride used to be `padded_m * padded_n`, which is one batch
    /// element. At `[3, 512, 512]` that put partial 1 at offset 262,144 —
    /// inside partial 0's batch 1 — so a batched split-K summed each
    /// partial into the next one's batch instead of into itself.
    #[test]
    fn split_k_scratch_slice_spans_the_whole_batch() {
        let facts = f32_facts(&[Dim::Const(3), Dim::Const(512), Dim::Const(512)]);
        let theta = SchedPoint::Coop {
            geom: geom(64, 64),
            splits: 4,
            staging: 2,
        };
        let layout = buffer_layout_for(&facts, Some(theta)).unwrap();
        assert_eq!(layout.rank(), 4);
        assert_eq!(layout.shape()[0], Dim::Const(4));
        assert_eq!(layout.strides()[0], Dim::Const(3 * 512 * 512));
        assert_eq!(layout_elements(&layout), Dim::Const(4 * 3 * 512 * 512));
        // ...and the partials do not overlap: slice `s` ends where `s+1`
        // begins.
        let Dim::Const(slice) = layout.strides()[0] else {
            panic!("a const shape must give a const slice stride");
        };
        let per_partial: u64 = layout.shape()[1..]
            .iter()
            .map(|d| d.as_const().expect("const"))
            .product();
        assert_eq!(slice, per_partial);
    }

    #[test]
    fn coop_padding_is_exact() {
        let facts = f32_facts(&[Dim::Const(100), Dim::Const(64)]);
        let theta = SchedPoint::Coop {
            geom: geom(64, 64),
            splits: 1,
            staging: 1,
        };
        let layout = buffer_layout_for(&facts, Some(theta)).unwrap();
        assert_eq!(layout.shape()[0], Dim::Const(128));
        assert_eq!(layout.shape()[1], Dim::Const(64));
        // The consumer reads a row every `padded_n` elements — exactly, not
        // "close enough": the reference's stride-equality test is this
        // invariant, established rather than checked.
        assert_eq!(layout.strides()[0], Dim::Const(64));
        assert_eq!(layout.strides()[1], Dim::Const(1));
    }

    #[test]
    fn a_symbolic_extent_stays_symbolic_through_padding() {
        let s = SymId(7);
        let facts = f32_facts(&[Dim::Sym(s), Dim::Const(64)]);
        let theta = SchedPoint::Coop {
            geom: geom(64, 64),
            splits: 1,
            staging: 1,
        };
        let layout = buffer_layout_for(&facts, Some(theta)).unwrap();
        assert_eq!(layout.shape()[0], Dim::Sym(s));
        assert_eq!(layout_elements(&layout), Dim::Sym(UNKNOWN_SYM));
    }

    #[test]
    fn contiguous_when_the_point_needs_no_padding() {
        let facts = f32_facts(&[Dim::Const(100), Dim::Const(37)]);
        let l = buffer_layout_for(&facts, Some(SchedPoint::Point)).unwrap();
        assert!(l.is_contiguous());
        assert_eq!(l.shape()[0], Dim::Const(100));
    }

    // -- PlanHash ---------------------------------------------------------

    /// `leaf[rows, 64] -> map -> map`, with `rows` supplied by the caller.
    fn family_graph(rows: Dim) -> (fusor2_ir::egraph::EGraph, Vec<Id>) {
        use crate::realize::testkit::{buffer, kmap, new_graph};
        let mut g = new_graph();
        let shape = [rows, Dim::Const(64)];
        let leaf = buffer(&mut g, 0, &shape);
        let a = kmap(&mut g, leaf, &shape, 1);
        let b = kmap(&mut g, a, &shape, 2);
        g.add_root(b);
        let roots = g.roots().to_vec();
        (g, roots)
    }

    fn hash_of(caps: fusor2_ir::device::Caps, rows: Dim) -> PlanHash {
        use crate::extract::LocalSearch;
        use fusor2_ir::extract::{ExtractBudget, Extractor};
        let (g, roots) = family_graph(rows);
        let cost = TestCost::with_caps(caps.clone());
        LocalSearch::new(std::sync::Arc::new(TestPlanner), caps)
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap()
            .hash
    }

    #[test]
    fn plan_hash_shape_family() {
        use crate::realize::testkit::test_caps;
        // One symbolic plan serves the whole family: nothing about the
        // binding is in the IR, so extracting it twice cannot differ.
        let sym = hash_of(test_caps(), Dim::Sym(SymId(0)));
        let sym_again = hash_of(test_caps(), Dim::Sym(SymId(0)));
        assert_eq!(sym, sym_again);

        // A *specialized* plan is a different plan, and says so.
        let small = hash_of(test_caps(), Dim::Const(128));
        let large = hash_of(test_caps(), Dim::Const(4096));
        assert_ne!(small, large);
        assert_ne!(sym, small);
    }

    #[test]
    fn plan_hash_includes_wg_storage() {
        use crate::realize::testkit::test_caps;
        let mut wide = test_caps();
        wide.limits.max_compute_workgroup_storage_size *= 2;
        assert_ne!(
            hash_of(test_caps(), Dim::Const(128)),
            hash_of(wide, Dim::Const(128)),
            "the coop legality filter reads this limit, so the plan must key on it"
        );
    }

    #[test]
    fn uniform_leaf_hashes_as_symbol() {
        use crate::extract::LocalSearch;
        use crate::realize::testkit::{buffer, new_graph, operand, test_caps};
        use fusor2_ir::extract::{ExtractBudget, Extractor};
        use fusor2_ir::ir::level0::LeafKind;
        use fusor2_ir::ir::level1::{IndexSpace, L1};
        use fusor2_ir::scalar::{BinOp, ScalarExpr};

        // Two graphs differing only in which symbol the runtime scalar is
        // bound through. Both land at index 0 of `symbols`, so both plans
        // are the same plan.
        let build = |sym: SymId| {
            let mut g = new_graph();
            let shape = [Dim::Const(1024)];
            let x = buffer(&mut g, 0, &shape);
            let u = g
                .add(Op::L0(L0::Leaf(LeafKind::Uniform {
                    sym,
                    dtype: Dtype::F32,
                })))
                .unwrap();
            let body = ScalarExpr::bin(
                BinOp::Mul,
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::uniform(sym, Dtype::F32),
            );
            let m = g
                .add(Op::L1(L1::KMap {
                    space: IndexSpace::new(shape),
                    body,
                    ops: vec![operand(x, &shape), operand(u, &[])],
                    sched: crate::realize::testkit::map_domain(),
                }))
                .unwrap();
            g.add_root(m);
            let roots = g.roots().to_vec();
            let cost = TestCost::default();
            LocalSearch::new(std::sync::Arc::new(TestPlanner), test_caps())
                .extract(&g, &roots, &cost, ExtractBudget::default())
                .unwrap()
        };
        let a = build(SymId(5));
        let b = build(SymId(9));
        assert_eq!(a.symbols.len(), 1);
        assert_eq!(a.symbols[0], SymId(5));
        assert_eq!(b.symbols[0], SymId(9));
        assert_eq!(
            a.hash, b.hash,
            "the uniform's slot is in the key; its identity is not"
        );
    }

    #[test]
    fn bindings_reserve_zero_for_the_uniform_block() {
        let (g, roots) = crate::realize::testkit::chain_graph(3);
        let ex = seed_for(&g, &roots);
        let cost = TestCost::default();
        let arena = TestPlanner;
        let r = crate::realize::realize(&g, &roots, &ex, &cost, &arena).unwrap();
        for c in &r.components {
            let b = derive_bindings(&g, &ex, &r, c).unwrap();
            assert!(b.iter().all(|x| x.binding >= 1));
            let mut idx: Vec<u32> = b.iter().map(|x| x.binding).collect();
            idx.sort_unstable();
            assert_eq!(idx, (1..=b.len() as u32).collect::<Vec<_>>());
            // reads first, then writes
            let first_write = b.iter().position(|x| x.kind != BindKind::Read);
            if let Some(w) = first_write {
                assert!(b[w..].iter().all(|x| x.kind != BindKind::Read));
            }
        }
    }
}
