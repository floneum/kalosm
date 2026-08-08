//! R3 and R4: the schedule domain attached to a contraction, the four
//! order-free family lowerings, and the epilogue un-fusing rule.
//!
//! `tile_contract` mints **one node, not four and not four hundred**: the
//! full legal `(geom x splits x staging)` space rides on the node and is
//! resolved by extraction. `family` is never stored on an L0 op, and
//! `ShapeSelector`'s first-match ordering is structurally impossible — all
//! four families are unioned into one chain unconditionally and compete on
//! cost.
//!
//! Owned by W4.

use fusor2_ir::contract_spec::partition;
use fusor2_ir::dtype::Dtype;
use fusor2_ir::egraph::{Builder, Facts, Id, RuleTag};
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::carrier::Carrier;
use fusor2_ir::ir::level0::{EinSpec, L0, Label};
use fusor2_ir::ir::level1::{
    AccessPlan, ContractSide, Family, IndexSpace, L1, Operand, ScheduleDomain,
};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;
use fusor2_ir::scalar::{BinOp, ScalarExpr, ScalarKind};
use fusor2_ir::shape::{Dim, Layout, SymId};
use smallvec::SmallVec;

use crate::domains::{
    DomainCtx, coop_domain, default_planner, fold_domain, map_domain, sgemm_domain, sgemv_domain,
};

rule!(
    TILE_CONTRACT,
    level = Level::L1,
    head = OpTag::KContract,
    tag = RuleTag::Additive,
    apply = tile_contract,
);

rule!(
    LOWER_COOP,
    level = Level::L0,
    head = OpTag::Contract,
    tag = RuleTag::StrictlyLowering,
    apply = lower_coop,
);

rule!(
    LOWER_SGEMM,
    level = Level::L0,
    head = OpTag::Contract,
    tag = RuleTag::StrictlyLowering,
    apply = lower_sgemm,
);

rule!(
    LOWER_SGEMV,
    level = Level::L0,
    head = OpTag::Contract,
    tag = RuleTag::StrictlyLowering,
    apply = lower_sgemv,
);

rule!(
    LOWER_GENERIC,
    level = Level::L0,
    head = OpTag::Contract,
    tag = RuleTag::StrictlyLowering,
    apply = lower_generic,
);

rule!(
    UNFUSE_COOP_EPILOGUE,
    level = Level::L1,
    head = OpTag::KContract,
    tag = RuleTag::Additive,
    apply = unfuse_coop_epilogue,
);

// ---------------------------------------------------------------------------
// Shared shape and expression helpers
// ---------------------------------------------------------------------------

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
        ScalarKind::Select { c, t, f } => {
            input_dtype(c).or_else(|| input_dtype(t)).or_else(|| input_dtype(f))
        }
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

/// Port of `coop_epilogues_supported` (`core/src/matmul/kernel.rs:152`),
/// verbatim: both pre chains dtype-preserving, the post chain reading the
/// operand dtype, and writing either the operand dtype or — for f16
/// operands — f32. A narrowing post would round the accumulator ahead of
/// the chain.
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
    let preserving = |e: &ScalarExpr| {
        e.dtype() == operand && input_dtype(e).is_none_or(|d| d == operand)
    };
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
        Op::L0(L0::Contract { spec, acc, a, b, .. }) => Some((spec, *acc, *a, *b)),
        _ => None,
    }
}

/// Whether this family can address these operands.
///
/// **A block-quantized operand is admitted to [`Family::Coop`] only.** The
/// cooperative kernel stages both operands into workgroup tiles before the MMA,
/// and `Source::Quantized` decodes at the `(row, col)` that staging fill
/// already computes — so a quantized weight is the format's decode math on the
/// way into shared memory and nothing else changes: same staging tile, same
/// fragments, same MMA, same arena footprint, same epilogue, same schedule
/// domain, same autotuner.
///
/// The other three families read operands element-wise from a dense layout with
/// no staging step to decode in, so they still decline. A quantized operand
/// they turn down still reaches a runnable form: `LOWER_DEQUANT` expands
/// `L0::Dequant` into its defn, and the contraction then sees a dense operand
/// like any other.
fn operands_addressable(f: &Facts<'_>, family: Family) -> bool {
    f.operands()
        .iter()
        .all(|o| !o.dtype.is_quantized() || family == Family::Coop)
}

/// The dtype the matrix unit actually sees. A quantized operand is decoded
/// during the staging fill, so the fragments are f32 and the storage format
/// never reaches the MMA — the coop legality probe and the MAC rate must be
/// asked about the decoded type, not the stored one.
fn compute_dtype(d: Dtype) -> Dtype {
    if d.is_quantized() { Dtype::F32 } else { d }
}

/// Whether the m/n/k kernels can address this spec's operands at all.
///
/// [`L1::KContract`] records four **extents** — `contract_mnk` takes the
/// products of the m, n, k and batch labels — and its operands are plain
/// contiguous layouts. That describes `a = [batch.., m.., k..]`,
/// `b = [batch.., k.., n..]`, `out = [batch.., m.., n..]` and nothing else,
/// so a spec in any other axis order is indistinguishable from the canonical
/// one at this level. `mat_mul_transposed_rhs` differs from `matmul` *only*
/// in that order, and the adjoint specs `d_lhs`/`d_rhs` are non-canonical by
/// construction, which is how `dB` came to read a `[m, k]` activation as if
/// it were `[k, m]`.
///
/// Declining leaves the contraction to `lower_contract_generic`, whose
/// operands carry the spec's geometry explicitly. That is slower and correct.
///
/// **The m/n/k families no longer read this.** Teaching `KContract` to carry
/// per-operand layouts — which is all `permuted_alias` does, since `Operand`
/// already holds a strided `Layout` — restored the fast path for exactly the
/// shapes this used to send to the floor, and it is where attention's cost
/// was: `q @ k^T` is `bhqd,bhkd->bhqk`, non-canonical in `b`, so Coop, SGEMM
/// and SGEMV all declined and the score matmul ran as a rank-5 generic reduce
/// beside a `p @ v` that got Coop. On `[1,8,1024,64]` that was **26.0 ms ->
/// 10.2 ms** end to end, with the score matmul becoming
/// `KContract{m:1024, n:1024, k:64, batch:8}`.
///
/// What still reads it is [`lower_generic`], whose `KFold` nest really does
/// address dense `[batch, m, k]` / `[batch, k, n]` aliases and has no operand
/// layout to permute. `out` is still required canonical everywhere, because
/// `KContract` does not parameterize its *write* map.
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
/// requiring the spec to have been written that way.
///
/// `L1::KContract` records four extents and addresses `a = [batch.., m.., k..]`,
/// `b = [batch.., k.., n..]`. That does not oblige the *buffer* to be stored in
/// that order — `Operand` carries a full strided `Layout`, and permuting the
/// strides states exactly the same read. This is what the architecture means by
/// "transposed-rhs is a spec, not an op": `mat_mul_transposed_rhs` differs from
/// `matmul` only in axis order, so it differs only in this stride vector.
///
/// Requiring canonical order instead is what sent every non-canonical
/// contraction to `lower_contract_generic`. Measured, and it is the whole of
/// attention's cost on this half of the chain: `q @ k^T` is
/// `bhqd,bhkd->bhqk`, whose `b` is `[batch, n, k]`, so **all three fast
/// families declined** and the score matmul ran as a rank-5 generic reduce
/// while `p @ v` next door — canonical — got `Family::Coop`. The adjoint specs
/// `d_lhs`/`d_rhs` are non-canonical by construction and were in the same
/// position.
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
    // `out` still has to be canonical: `KContract` does not parameterize its
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
        // A generic-fold contraction *is* a fold, so it takes the fold's
        // domain rather than a point: reduction strategy and lane-group width
        // are the same late decision here as at any other reduction.
        Family::GenericFold => ScheduleDomain::Fold(fold_domain(mnk.k, &cx)),
    };

    let op = L1::KContract {
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
    let new = b.add_l1(op).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

// ---------------------------------------------------------------------------
// The rules
// ---------------------------------------------------------------------------

/// Attach the complete legal schedule domain to a `KContract` that arrived
/// carrying [`ScheduleDomain::Point`]. An empty domain means the rule does
/// not apply, never that the node is broken.
pub fn tile_contract(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(l1) = &node.op else { return None };
    let L1::KContract {
        m,
        n,
        k,
        batch,
        family,
        acc,
        sched: ScheduleDomain::Point,
        ..
    } = l1
    else {
        return None;
    };
    let operand = f.dtype(0).unwrap_or(*acc);
    let cx = DomainCtx::new(f.caps(), default_planner());

    let sched = match family {
        Family::Coop => {
            let dom = coop_domain(*m, *n, *k, *batch, operand, *acc, &cx);
            if dom.is_empty() {
                return None;
            }
            ScheduleDomain::Coop(dom)
        }
        Family::Sgemm => {
            let dom = sgemm_domain(operand.byte_size() as u32, &cx);
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
        Family::GenericFold => {
            let dom = fold_domain(*k, &cx);
            if dom.strategies.is_empty() {
                return None;
            }
            ScheduleDomain::Fold(dom)
        }
    };

    let mut rebuilt = l1.clone();
    if let L1::KContract { sched: s, .. } = &mut rebuilt {
        *s = sched;
    }
    let new = b.add_l1(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// `Contract -> KContract { family: Coop }`. Guarded on a *fixed* subgroup
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

/// `Contract -> KContract { family: Sgemm }`. Unguarded on shape.
pub fn lower_sgemm(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    lower_family(b, id, node, f, Family::Sgemm)
}

/// `Contract -> KContract { family: Sgemv }`. Unguarded on shape; a badly
/// shaped gemv simply loses on cost.
pub fn lower_sgemv(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    lower_family(b, id, node, f, Family::Sgemv)
}

/// `Contract -> KFold` at an `Add` carrier lifting `mul(Arg0, Arg1)`. The floor
/// that guarantees every contraction reaches a runnable form.
pub fn lower_generic(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    // The generic-fold floor reads its operands element-wise from a dense
    // layout and has no staging step to decode a block format in, so a
    // quantized operand is left to `LOWER_DEQUANT`, which expands the decode
    // into an ordinary `Map` this rule can then read.
    if !operands_addressable(f, Family::GenericFold) {
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
    let pre = ScalarExpr::bin(
        BinOp::Mul,
        ScalarExpr::arg(0, acc),
        ScalarExpr::arg(1, acc),
    );
    let op = L1::KFold {
        space,
        axis: 3,
        vec_axes: smallvec::SmallVec::new(),
        carrier: Carrier::binop(BinOp::Add, Carrier::binop_identity(BinOp::Add, acc)?, acc)
            .with_lift([pre]),
        acc,
        post: smallvec::smallvec![identity(acc)],
        ops: vec![alias(a_id, fa), alias(b_id, fb)],
        sched: ScheduleDomain::Fold(fold_domain(mnk.k, &cx)),
    };
    let new = b.add_l1(op).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// Split an epilogue the cooperative kernel cannot host into
/// `KMap{body: post} . KContract{post: identity}`.
///
/// A hostable epilogue fires no rule; `lower_generic`'s node is already the
/// third alternative. This is what stops one unsupported activation from
/// costing the whole coop speedup, which is what the reference's
/// route-by-refusal does.
pub fn unfuse_coop_epilogue(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
) -> Option<Id> {
    let Op::L1(l1) = &node.op else { return None };
    let L1::KContract {
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
    if let L1::KContract { post: p, .. } = &mut inner_op {
        *p = identity(acc);
    }
    let inner = b.add_l1(inner_op).ok()?;

    let shape = [batch, m, n];
    let cx = DomainCtx::new(f.caps(), default_planner());
    let inner_facts = b.facts_of(inner).clone();
    let outer = b
        .add_l1(L1::KMap {
            space: IndexSpace::new(shape),
            body: post,
            ops: vec![alias(inner, &inner_facts)],
            sched: ScheduleDomain::Map(map_domain(&shape, &[AccessPlan::Alias], &cx)),
        })
        .ok()?;
    b.union(id, outer).ok()?;
    Some(outer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::testing::{apple_caps, no_coop_caps};
    use crate::rules::TILE_RULES;
    use crate::rules::testing::{Fixture, l1_of};
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::ir::level0::EinSpec;
    use smallvec::smallvec;

    fn matmul_spec() -> EinSpec {
        EinSpec {
            a: smallvec![Label(b'm'), Label(b'k')],
            b: smallvec![Label(b'k'), Label(b'n')],
            out: smallvec![Label(b'm'), Label(b'n')],
        }
    }

    #[test]
    fn mnk_splits_a_batched_spec() {
        let spec = EinSpec {
            a: smallvec![Label(b'b'), Label(b'm'), Label(b'k')],
            b: smallvec![Label(b'b'), Label(b'k'), Label(b'n')],
            out: smallvec![Label(b'b'), Label(b'm'), Label(b'n')],
        };
        let a = ValueFacts::new(Dtype::F32, [Dim::Const(4), Dim::Const(32), Dim::Const(16)]);
        let b = ValueFacts::new(Dtype::F32, [Dim::Const(4), Dim::Const(16), Dim::Const(8)]);
        assert_eq!(
            contract_mnk(&spec, &a, &b),
            Mnk {
                m: Dim::Const(32),
                n: Dim::Const(8),
                k: Dim::Const(16),
                batch: Dim::Const(4),
            }
        );
    }

    #[test]
    fn all_four_families_coexist() {
        let caps = apple_caps();
        let mut fx = Fixture::new(caps);
        let a = fx.buffer(Dtype::F32, &[4096, 4096]);
        let b = fx.buffer(Dtype::F32, &[4096, 4096]);
        let c = fx.contract(matmul_spec(), Dtype::F32, a, b);
        fx.apply_all(TILE_RULES, c);

        let mut coop = 0;
        let mut sgemm = 0;
        let mut sgemv = 0;
        let mut folds = 0;
        for m in fx.chain(c) {
            match l1_of(&fx, m) {
                Some(L1::KContract { family, .. }) => match family {
                    Family::Coop => coop += 1,
                    Family::Sgemm => sgemm += 1,
                    Family::Sgemv => sgemv += 1,
                    Family::GenericFold => {}
                },
                Some(L1::KFold { .. }) => folds += 1,
                _ => {}
            }
        }
        assert_eq!((coop, sgemm, sgemv, folds), (1, 1, 1, 1));
    }

    #[test]
    fn no_coop_without_caps() {
        let caps = no_coop_caps();
        let mut fx = Fixture::new(caps);
        let a = fx.buffer(Dtype::F32, &[1024, 1024]);
        let b = fx.buffer(Dtype::F32, &[1024, 1024]);
        let c = fx.contract(matmul_spec(), Dtype::F32, a, b);
        fx.apply_all(TILE_RULES, c);

        let members: Vec<L1> = fx.chain(c).into_iter().filter_map(|m| l1_of(&fx, m)).collect();
        assert_eq!(members.len(), 3);
        assert!(!members.iter().any(|m| matches!(
            m,
            L1::KContract {
                family: Family::Coop,
                ..
            }
        )));
    }

    #[test]
    fn the_coop_node_carries_the_whole_domain() {
        let caps = apple_caps();
        let mut fx = Fixture::new(caps);
        let a = fx.buffer(Dtype::F32, &[4096, 4096]);
        let b = fx.buffer(Dtype::F32, &[4096, 4096]);
        let c = fx.contract(matmul_spec(), Dtype::F32, a, b);
        fx.apply_all(TILE_RULES, c);

        let coop = fx
            .chain(c)
            .into_iter()
            .filter_map(|m| l1_of(&fx, m))
            .find(|m| {
                matches!(
                    m,
                    L1::KContract {
                        family: Family::Coop,
                        ..
                    }
                )
            })
            .expect("a coop alternative");
        let L1::KContract { sched, .. } = &coop else {
            unreachable!()
        };
        assert!(sched.len() > 1000, "domain has {} points", sched.len());
    }

    #[test]
    fn unfuse_fires_on_narrowing_post() {
        // f32 operands with an f32 -> f16 post: the coop store would round
        // the accumulator ahead of the chain, so the pair is minted.
        let caps = apple_caps();
        let mut fx = Fixture::new(caps.clone());
        let a = fx.buffer(Dtype::F32, &[512, 512]);
        let b = fx.buffer(Dtype::F32, &[512, 512]);
        let c = fx.contract(matmul_spec(), Dtype::F32, a, b);
        fx.apply_all(TILE_RULES, c);
        let coop = fx
            .chain(c)
            .into_iter()
            .find(|m| {
                matches!(
                    l1_of(&fx, *m),
                    Some(L1::KContract {
                        family: Family::Coop,
                        ..
                    })
                )
            })
            .expect("a coop alternative");
        let narrowed = fx.with_post(coop, ScalarExpr::cast(Dtype::F16, ScalarExpr::arg(0, Dtype::F32)));
        fx.apply_all(TILE_RULES, narrowed);
        assert!(
            fx.chain(narrowed)
                .into_iter()
                .any(|m| matches!(l1_of(&fx, m), Some(L1::KMap { .. }))),
            "a narrowing post must mint the un-fused pair"
        );

        // f16 operands with an f16 -> f32 post: the fused matmul-then-cast
        // mixed-precision training emits for every weight gradient. Hosted,
        // so nothing is minted.
        let mut fx = Fixture::new(caps);
        let a = fx.buffer(Dtype::F16, &[512, 512]);
        let b = fx.buffer(Dtype::F16, &[512, 512]);
        let c = fx.contract(matmul_spec(), Dtype::F16, a, b);
        fx.apply_all(TILE_RULES, c);
        let coop = fx
            .chain(c)
            .into_iter()
            .find(|m| {
                matches!(
                    l1_of(&fx, *m),
                    Some(L1::KContract {
                        family: Family::Coop,
                        ..
                    })
                )
            })
            .expect("a coop alternative");
        let widened = fx.with_post(coop, ScalarExpr::cast(Dtype::F32, ScalarExpr::arg(0, Dtype::F16)));
        fx.apply_all(TILE_RULES, widened);
        assert!(
            !fx.chain(widened)
                .into_iter()
                .any(|m| matches!(l1_of(&fx, m), Some(L1::KMap { .. }))),
            "a hostable epilogue must fire no rule"
        );
    }

    #[test]
    fn hostable_epilogue_predicate_matches_the_reference() {
        let f32_id = identity(Dtype::F32);
        let f16_id = identity(Dtype::F16);
        assert!(coop_epilogue_hostable(&f32_id, &f32_id, &f32_id, Dtype::F32));
        // f16 -> f32 post is the one widening the store may host.
        let widen = ScalarExpr::cast(Dtype::F32, ScalarExpr::arg(0, Dtype::F16));
        assert!(coop_epilogue_hostable(&f16_id, &f16_id, &widen, Dtype::F16));
        // f32 -> f16 is a narrowing and is not.
        let narrow = ScalarExpr::cast(Dtype::F16, ScalarExpr::arg(0, Dtype::F32));
        assert!(!coop_epilogue_hostable(&f32_id, &f32_id, &narrow, Dtype::F32));
        // A narrowing *pre* chain is refused too.
        assert!(!coop_epilogue_hostable(&narrow, &f32_id, &f32_id, Dtype::F32));
    }

    #[test]
    fn tile_contract_upgrades_a_point_scheduled_node() {
        let caps = apple_caps();
        let mut fx = Fixture::new(caps);
        let a = fx.buffer(Dtype::F32, &[1024, 1024]);
        let b = fx.buffer(Dtype::F32, &[1024, 1024]);
        let point = fx.point_contract(Family::Coop, Dtype::F32, a, b, 1024);
        fx.apply_all(TILE_RULES, point);
        let upgraded = fx
            .chain(point)
            .into_iter()
            .filter_map(|m| l1_of(&fx, m))
            .any(|m| matches!(m, L1::KContract { sched, .. } if sched.len() > 1));
        assert!(upgraded, "tile_contract must attach the domain");
    }
}
