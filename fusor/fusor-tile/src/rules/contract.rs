//! R3 and R4: the order-free family lowerings and the epilogue un-fusing
//! rule.
//!
//! `lower_family` mints **one node, not four and not four hundred**: the
//! full legal `(geom x splits x staging)` space rides on the node and is
//! resolved by extraction.

use fusor_ir::carrier::Carrier;
use fusor_ir::contract_spec::partition;
use fusor_ir::dtype::Dtype;
use fusor_ir::egraph::{Builder, Facts, Id, RuleTag};
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::launch::{
    AccessPlan, ContractSide, Family, IndexSpace, Launch, Operand, ScheduleDomain,
};
use fusor_ir::ir::logical::{EinSpec, Label, Logical};
use fusor_ir::ir::{Level, Node, Op, OpTag};
use fusor_ir::rule;
use fusor_ir::scalar::{BinOp, ScalarExpr, ScalarKind};
use fusor_ir::shape::{Dim, Layout, SymId};
use smallvec::SmallVec;

use crate::domains::{
    DomainCtx, coop_domain, default_planner, fold_domain, map_domain, sgemm_domain, sgemv_domain,
};

rule!(
    LOWER_COOP,
    level = Level::Logical,
    head = OpTag::Contract,
    tag = RuleTag::StrictlyLowering,
    apply = lower_coop,
);

rule!(
    LOWER_SGEMM,
    level = Level::Logical,
    head = OpTag::Contract,
    tag = RuleTag::StrictlyLowering,
    apply = lower_sgemm,
);

rule!(
    LOWER_SGEMV,
    level = Level::Logical,
    head = OpTag::Contract,
    tag = RuleTag::StrictlyLowering,
    apply = lower_sgemv,
);

rule!(
    LOWER_GENERIC,
    level = Level::Logical,
    head = OpTag::Contract,
    tag = RuleTag::StrictlyLowering,
    apply = lower_generic,
);

rule!(
    UNFUSE_COOP_EPILOGUE,
    level = Level::Launch,
    head = OpTag::LaunchContract,
    tag = RuleTag::Additive,
    apply = unfuse_coop_epilogue,
);

/// The four extents a contraction kernel is parameterized by.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Mnk {
    pub m: Dim,
    pub n: Dim,
    pub k: Dim,
    pub batch: Dim,
}

/// An extent that could not be folded to a constant because two or more
/// symbolic labels multiply into it. Mirrors `Layout::row_major_strides`,
/// which reaches for the same opaque symbol for the same reason.
const OPAQUE: Dim = Dim::Sym(SymId(u32::MAX));

fn dim_product(dims: &[Dim]) -> Dim {
    let mut acc: u64 = 1;
    let mut symbolic: Option<Dim> = None;
    for d in dims {
        match d {
            Dim::Const(v) => acc = acc.saturating_mul(*v),
            Dim::Sym(_) => {
                if symbolic.is_some() {
                    return OPAQUE;
                }
                symbolic = Some(*d);
            }
        }
    }
    match symbolic {
        None => Dim::Const(acc),
        Some(s) if acc == 1 => s,
        Some(_) => OPAQUE,
    }
}

fn extent(labels: &[Label], shape: &[Dim], want: Label) -> Option<Dim> {
    labels
        .iter()
        .position(|l| *l == want)
        .and_then(|i| shape.get(i).copied())
}

/// Split an [`EinSpec`] into the `(m, n, k, batch)` a kernel launches over.
/// A label in `a` and `b` but not `out` is summed; one in all three is a
/// batch axis; one in `a`/`out` only is an `m` axis and its mirror an `n`.
pub fn contract_mnk(spec: &EinSpec, a: &ValueFacts, b: &ValueFacts) -> Mnk {
    let (mut ms, mut ns, mut ks, mut batches) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for label in spec.a.iter().copied() {
        let Some(d) = extent(&spec.a, &a.shape, label) else {
            continue;
        };
        let in_b = spec.b.contains(&label);
        let in_out = spec.out.contains(&label);
        match (in_b, in_out) {
            (true, true) => batches.push(d),
            (true, false) => ks.push(d),
            (false, _) => ms.push(d),
        }
    }
    for label in spec.b.iter().copied() {
        if spec.a.contains(&label) {
            continue;
        }
        if let Some(d) = extent(&spec.b, &b.shape, label) {
            ns.push(d);
        }
    }
    Mnk {
        m: dim_product(&ms),
        n: dim_product(&ns),
        k: dim_product(&ks),
        batch: dim_product(&batches),
    }
}

/// `Arg(0)` — the epilogue that does nothing.
pub fn identity(dtype: Dtype) -> ScalarExpr {
    ScalarExpr::arg(0, dtype)
}

pub fn is_identity(e: &ScalarExpr) -> bool {
    matches!(e.kind(), ScalarKind::Arg(0))
}

/// The dtype the first `Arg` leaf of an expression reads.
pub fn input_dtype(e: &ScalarExpr) -> Option<Dtype> {
    match e.kind() {
        ScalarKind::Arg(_) => Some(e.dtype()),
        ScalarKind::Lit(_) | ScalarKind::Uniform(_) | ScalarKind::IndexOf(_) => None,
        ScalarKind::Un { x, .. }
        | ScalarKind::Cast { x, .. }
        | ScalarKind::Bitcast { x, .. }
        | ScalarKind::Round { x, .. }
        | ScalarKind::Splat { x, .. } => input_dtype(x),
        ScalarKind::Bin { a, b, .. } | ScalarKind::Cmp { a, b, .. } | ScalarKind::Dot { a, b } => {
            input_dtype(a).or_else(|| input_dtype(b))
        }
        ScalarKind::Select { c, t, f } => input_dtype(c)
            .or_else(|| input_dtype(t))
            .or_else(|| input_dtype(f)),
    }
}

/// An aliasing operand over a value's own contiguous layout.
pub fn alias(src: Id, facts: &ValueFacts) -> Operand {
    Operand {
        src,
        layout: Layout::contiguous(&facts.shape),
        access: AccessPlan::Alias,
    }
}

/// Both pre chains must be dtype-preserving, the post chain must read the
/// operand dtype and write either the operand dtype or — for f16 operands —
/// f32. A narrowing post would round the accumulator ahead of the chain.
///
/// This is a *legality* predicate on the epilogue, never a routing
/// decision: an epilogue the coop kernel cannot host un-fuses into a second
/// dispatch as one alternative and routes to the generic fold as another.
pub fn coop_epilogue_hostable(
    pre_a: &ScalarExpr,
    pre_b: &ScalarExpr,
    post: &ScalarExpr,
    operand: Dtype,
) -> bool {
    let preserving =
        |e: &ScalarExpr| e.dtype() == operand && input_dtype(e).is_none_or(|d| d == operand);
    if !preserving(pre_a) || !preserving(pre_b) {
        return false;
    }
    if input_dtype(post).is_some_and(|d| d != operand) {
        return false;
    }
    post.dtype() == operand || (operand == Dtype::F16 && post.dtype() == Dtype::F32)
}

fn contract_parts(node: &Node) -> Option<(&EinSpec, Dtype, Id, Id)> {
    match &node.op {
        Op::Logical(Logical::Contract {
            spec, acc, a, b, ..
        }) => Some((spec, *acc, *a, *b)),
        _ => None,
    }
}

/// Whether this family can address these operands.
///
/// A block-quantized operand is admitted to [`Family::Coop`] (which decodes
/// during the staging fill into workgroup tiles) and, on GPU, to
/// [`Family::Sgemv`] (which decodes per loaded element). The other families
/// read operands element-wise from a dense layout with no staging step to
/// decode in, so they decline; `LOWER_DEQUANT` still gets those operands to a
/// runnable form.
fn operands_addressable(f: &Facts<'_>, family: Family) -> bool {
    // The CPU nest reads buffers through plain collapsed strides and would
    // read block words as floats, so it keeps the dense-only rule.
    let q_ok = |family: Family| {
        family == Family::Coop
            || (family == Family::Sgemv && f.caps().kind == fusor_ir::device::DeviceKind::Gpu)
    };
    f.operands()
        .iter()
        .all(|o| !o.dtype.is_quantized() || q_ok(family))
}

/// The dtype the matrix unit actually sees. A quantized operand is decoded
/// during the staging fill, so the fragments are f32 and the storage format
/// never reaches the MMA — the coop legality probe and the MAC rate must be
/// asked about the decoded type, not the stored one.
fn compute_dtype(d: Dtype) -> Dtype {
    if d.is_quantized() { Dtype::F32 } else { d }
}

/// Whether [`lower_generic`]'s `Fold` nest can address this spec: its
/// operands are dense `[batch, m, k]` / `[batch, k, n]` aliases with no
/// operand layout to permute, and [`Launch::Contract`] records extents only,
/// so a spec in any other axis order is indistinguishable from the canonical
/// one at this level. Declining leaves the contraction to
/// `lower_contract_generic`, whose operands carry the spec's geometry
/// explicitly. `out` is required canonical everywhere, because `Contract`
/// does not parameterize its *write* map.
fn canonical_for_mnk(spec: &EinSpec) -> bool {
    let Ok(part) = partition(spec) else {
        return false;
    };
    let cat = |groups: [&[Label]; 2]| -> SmallVec<[Label; 8]> {
        let mut v: SmallVec<[Label; 8]> = SmallVec::new();
        for g in groups {
            v.extend(g.iter().copied());
        }
        v
    };
    let want_a = cat([&part.batch, &part.m]);
    let want_b = cat([&part.batch, &part.k]);
    let want_out = cat([&part.batch, &part.m]);
    spec.a[..] == cat([&want_a, &part.k])[..]
        && spec.b[..] == cat([&want_b, &part.n])[..]
        && spec.out[..] == cat([&want_out, &part.n])[..]
}

/// Read an operand in canonical label order **through its layout** rather than
/// requiring the spec to have been written that way: `Operand` carries a full
/// strided `Layout`, and permuting the strides states exactly the same read.
///
/// Returns `None` when a label is missing or an extent is symbolic, in which
/// case the caller declines and the generic fold still carries the value.
fn permuted_alias(
    src: Id,
    facts: &ValueFacts,
    actual: &[Label],
    want: &[Label],
) -> Option<Operand> {
    if actual.len() != facts.shape.len() || want.len() != actual.len() {
        return None;
    }
    let strides = Layout::row_major_strides(&facts.shape);
    let mut shape: SmallVec<[Dim; 6]> = SmallVec::new();
    let mut perm: SmallVec<[Dim; 6]> = SmallVec::new();
    for l in want {
        let i = actual.iter().position(|x| x == l)?;
        shape.push(facts.shape[i]);
        perm.push(strides[i]);
    }
    Some(Operand {
        src,
        layout: Layout::from_parts(Dim::Const(0), &shape, &perm).ok()?,
        access: AccessPlan::Alias,
    })
}

fn lower_family(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
    family: Family,
) -> Option<Id> {
    if !operands_addressable(f, family) {
        return None;
    }
    let (spec, acc, a_id, b_id) = contract_parts(node)?;
    let (fa, fb) = (f.operand(0)?, f.operand(1)?);
    let operand_dtype = compute_dtype(fa.dtype);
    // `out` still has to be canonical: `Contract` does not parameterize its
    // *write* map, so an output in another axis order genuinely is a different
    // kernel. Both *reads* are a stride vector away from canonical, so they are
    // permuted rather than required.
    let part = partition(spec).ok()?;
    let cat = |x: &[Label], y: &[Label]| -> SmallVec<[Label; 8]> {
        x.iter().chain(y.iter()).copied().collect()
    };
    let want_a = cat(&cat(&part.batch, &part.m), &part.k);
    let want_b = cat(&cat(&part.batch, &part.k), &part.n);
    let want_out = cat(&cat(&part.batch, &part.m), &part.n);
    if spec.out[..] != want_out[..] {
        return None;
    }
    let a_op = permuted_alias(a_id, fa, &spec.a, &want_a)?;
    let b_op = permuted_alias(b_id, fb, &spec.b, &want_b)?;
    let mnk = contract_mnk(spec, fa, fb);
    let cx = DomainCtx::new(f.caps(), default_planner());

    let sched = match family {
        Family::Coop => {
            let dom = coop_domain(mnk.m, mnk.n, mnk.k, mnk.batch, operand_dtype, acc, &cx);
            if dom.is_empty() {
                return None;
            }
            ScheduleDomain::Coop(dom)
        }
        Family::Sgemm => {
            let dom = sgemm_domain(operand_dtype.byte_size() as u32, &cx);
            if dom.params.is_empty() {
                return None;
            }
            ScheduleDomain::Sgemm(dom)
        }
        Family::Sgemv => {
            let dom = sgemv_domain(&cx);
            if dom.params.is_empty() {
                return None;
            }
            ScheduleDomain::Sgemv(dom)
        }
    };

    let op = Launch::Contract {
        m: mnk.m,
        n: mnk.n,
        k: mnk.k,
        batch: mnk.batch,
        family,
        post: identity(acc),
        acc,
        a: ContractSide::one(identity(compute_dtype(fa.dtype)), a_op),
        b: ContractSide::one(identity(compute_dtype(fb.dtype)), b_op),
        sched,
    };
    let new = b.add_launch(op).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// `Contract -> Contract { family: Coop }`. Guarded on a *fixed* subgroup
/// width, a reported cooperative configuration for this `(operand, acc)`
/// pair, and a non-empty legal geometry set. Legality only: a badly padded
/// coop tile is still a candidate and loses on cost.
pub fn lower_coop(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let (_, acc, _, _) = contract_parts(node)?;
    let operand = f.dtype(0)?;
    if !f.caps().coop_supported() || f.caps().coop_for(operand, acc).is_none() {
        return None;
    }
    lower_family(b, id, node, f, Family::Coop)
}

/// `Contract -> Contract { family: Sgemm }`. Unguarded on shape.
pub fn lower_sgemm(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    lower_family(b, id, node, f, Family::Sgemm)
}

/// `Contract -> Contract { family: Sgemv }`. Unguarded on shape; a badly
/// shaped gemv simply loses on cost.
pub fn lower_sgemv(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    lower_family(b, id, node, f, Family::Sgemv)
}

/// `Contract -> Fold` at an `Add` carrier lifting `mul(Arg0, Arg1)`. The floor
/// that guarantees every contraction reaches a runnable form.
pub fn lower_generic(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    // The generic-fold floor reads its operands element-wise from a dense
    // layout and has no staging step to decode a block format in, so a
    // quantized operand is left to `LOWER_DEQUANT`, which expands the decode
    // into an ordinary `Map` this rule can then read.
    if f.operands().iter().any(|o| o.dtype.is_quantized()) {
        return None;
    }
    let (spec, acc, a_id, b_id) = contract_parts(node)?;
    // Same restriction as the m/n/k families: this nest's operands are dense
    // aliases over `[batch, m, k]` and `[batch, k, n]`, so it cannot address a
    // spec whose axes sit in another order. `lower_contract_generic` can.
    if !canonical_for_mnk(spec) {
        return None;
    }
    let (fa, fb) = (f.operand(0)?, f.operand(1)?);
    let mnk = contract_mnk(spec, fa, fb);
    let cx = DomainCtx::new(f.caps(), default_planner());

    let space = IndexSpace::new([mnk.batch, mnk.m, mnk.n, mnk.k]);
    // The operands are stated over the fold's own index space as stride-0
    // broadcast views: A is `[batch, m, n, k]` with strides `[m*k, k, 0, 1]`,
    // B with `[k*n, 0, 1, n]`. Every lowering addresses a `Fold`'s operands
    // through their own layout maps, so an operand stated over any other
    // space is read at garbage addresses.
    //
    // The broadcast strides need constant `m`, `n`, `k`; a symbolic extent
    // has no spellable stride product, and a node whose operands cannot be
    // stated over its space must not be minted at all. The m/n/k
    // `Contract` families keep the floor for those shapes.
    let (m, n, k) = (mnk.m.as_const()?, mnk.n.as_const()?, mnk.k.as_const()?);
    let dims4 = [mnk.batch, mnk.m, mnk.n, mnk.k];
    let strided = |strides: [u64; 4]| -> Option<Layout> {
        Layout::from_parts(Dim::Const(0), &dims4, &strides.map(Dim::Const)).ok()
    };
    let a_op = Operand {
        src: a_id,
        layout: strided([m * k, k, 0, 1])?,
        access: AccessPlan::Alias,
    };
    let b_op = Operand {
        src: b_id,
        layout: strided([k * n, 0, 1, n])?,
        access: AccessPlan::Alias,
    };
    let pre = ScalarExpr::bin(BinOp::Mul, ScalarExpr::arg(0, acc), ScalarExpr::arg(1, acc));
    let op = Launch::Fold {
        space,
        axis: 3,
        vec_axes: smallvec::SmallVec::new(),
        carrier: Carrier::binop(BinOp::Add, Carrier::binop_identity(BinOp::Add, acc)?, acc)
            .with_lift([pre]),
        acc,
        post: smallvec::smallvec![identity(acc)],
        ops: vec![a_op, b_op],
        sched: ScheduleDomain::Fold(fold_domain(mnk.k, &cx)),
    };
    let new = b.add_launch(op).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// Split an epilogue the cooperative kernel cannot host into
/// `Map{body: post} . Contract{post: identity}`.
///
/// A hostable epilogue fires no rule; `lower_generic`'s node is already the
/// third alternative. This is what stops one unsupported activation from
/// costing the whole coop speedup.
pub fn unfuse_coop_epilogue(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(l1) = &node.op else {
        return None;
    };
    let Launch::Contract {
        m,
        n,
        batch,
        family: Family::Coop,
        a,
        b: rhs,
        post,
        acc,
        ..
    } = l1
    else {
        return None;
    };
    if is_identity(post) {
        return None;
    }
    let operand = f.dtype(0).unwrap_or(*acc);
    if coop_epilogue_hostable(&a.pre, &rhs.pre, post, operand) {
        return None;
    }

    let (post, m, n, batch, acc) = (post.clone(), *m, *n, *batch, *acc);
    let mut inner_op = l1.clone();
    if let Launch::Contract { post: p, .. } = &mut inner_op {
        *p = identity(acc);
    }
    let inner = b.add_launch(inner_op).ok()?;

    let shape = [batch, m, n];
    let cx = DomainCtx::new(f.caps(), default_planner());
    let inner_facts = b.facts_of(inner).clone();
    let outer = b
        .add_launch(Launch::Map {
            space: IndexSpace::new(shape),
            body: post,
            ops: vec![alias(inner, &inner_facts)],
            sched: ScheduleDomain::Map(map_domain(&shape, &[AccessPlan::Alias], &cx)),
        })
        .ok()?;
    b.union(id, outer).ok()?;
    Some(outer)
}
