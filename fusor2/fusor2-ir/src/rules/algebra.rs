//! L0 algebra: the fold-splitting law, additive contraction recognition,
//! contraction reassociation, closed-expression folding, identity
//! elimination, the mixed-precision store cast and the unit-fold collapse.
//!
//! Every rule here is `Additive`: the unrewritten form stays live in the same
//! chain, and which one runs is decided once, later, by extraction.

use crate::dtype::{Dtype, RoundMode, Splat};
use crate::egraph::{Builder, Facts, Id, RuleTag};
#[cfg(test)]
use crate::carrier::Carrier;
use crate::ir::level0::{EinSpec, L0, Label, LeafKind};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::scalar::{BinOp, CmpOp, Lit, ScalarExpr, ScalarKind, UnOp};
use crate::shape::{BoundsProof, Dim, StrideSpec};
use smallvec::SmallVec;

rule!(
    STRIP,
    level = Level::L0,
    head = OpTag::Fold,
    tag = RuleTag::Additive,
    apply = strip,
);

rule!(
    RECOGNIZE_CONTRACT,
    level = Level::L0,
    head = OpTag::Fold,
    tag = RuleTag::Additive,
    apply = recognize_contract,
);

rule!(
    CONTRACT_REASSOC,
    level = Level::L0,
    head = OpTag::Contract,
    tag = RuleTag::Additive,
    apply = contract_reassoc,
);

rule!(
    CONST_FOLD_MAP,
    level = Level::L0,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = const_fold_map,
);

rule!(
    IDENTITY_ELIM,
    level = Level::L0,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = identity_elim,
);

rule!(
    WIDEN_STORE_CAST,
    level = Level::L0,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = widen_store_cast,
);

rule!(
    UNIT_FOLD_COLLAPSE,
    level = Level::L0,
    head = OpTag::Fold,
    tag = RuleTag::Additive,
    l0 = Fold {
        carrier,
        axis,
        ins
    },
    |b, id, node, f| {
        let _ = node;
        // Only a single scalar slot whose lift is the bare element: a lift
        // that computes anything still has to run, and a multi-slot carrier's
        // output carries an axis that dropping the fold would delete.
        if carrier.width() != 1
            || carrier.slots[0] != crate::carrier::SlotTy::Scalar
            || carrier.lift[0].kind() != &ScalarKind::Arg(0)
        {
            return None;
        }
        let &[x] = &ins[..] else {
            return None;
        };
        let shape = &f.operand(0)?.shape;
        let axis = *axis as usize;
        if axis >= shape.len() || !shape[axis].known_eq(Dim::ONE) {
            return None;
        }
        // Combining a single element with the identity is that element, so
        // dropping the axis is the whole rewrite.
        let specs: SmallVec<[StrideSpec; 6]> = (0..shape.len())
            .filter(|&j| j != axis)
            .map(|j| StrideSpec::dim(j as u32, shape[j]))
            .collect();
        let dropped = b
            .add_l0(L0::Restride {
                specs,
                bounds: BoundsProof::Static,
                x,
            })
            .ok()?;
        b.union(id, dropped).ok()
    },
);

// ---------------------------------------------------------------------------
// STRIP — two clauses over one object, the reduction domain
// ---------------------------------------------------------------------------

/// STRIP. Both clauses are about the reduction domain and both are minted at
/// the same node, because the driver's fired set is per `(RuleId, Id)`.
///
/// * **SPLIT** — a catamorphism over a concatenation is the merge of the
///   catamorphisms over the segments.
/// * **ELIDE** — a block whose lift is identically the carrier's identity
///   contributes nothing, because `merge(acc, identity) = acc`.
pub fn strip(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    // ELIDE first: narrowing the domain makes the split cheaper, and both
    // alternatives stay live in the same class either way.
    let elided = fold_elide(b, node, f).and_then(|x| b.union(id, x).ok());
    let split = fold_split(b, id, node, f);
    split.or(elided)
}

/// SPLIT: `Fold{c, axis}` == `Fold{c.as_merge(), axis} . Fold{c, axis+1} . block(x)`.
///
/// At a contraction's summed axis this *is* split-K; at a `(max, sum)`
/// carrier it *is* online softmax; at `(n, mean, m2)` it *is* the stable
/// variance accumulator — all three fall out with no extra code, because the
/// carrier rides through untouched and this rule never reads what it means.
///
/// The outer level uses [`Carrier::as_merge`]: its elements are partial
/// **accumulators**, not raw elements, and it reads ONE operand carrying the
/// inner fold's trailing carrier axis rather than `width` operands. Reusing
/// the inner carrier applies `lift` to a partial max and silently computes a
/// wrong value; at a single-slot binop the two spellings coincide, which is
/// how that bug survived.
///
/// The `reassoc` guard is load-bearing. Without it the split and unsplit
/// forms are declared value-equal, and extraction swaps them on an f16
/// accumulator, in a system whose acceptance test is a byte-identical QAT
/// export.
///
/// **The `at_least(4096)` gate is gone**, and with it the reason there was no
/// split-K and no block loop at any length the trainer or the conformance
/// matmul and attention cases use. What replaces it is smaller and is a device
/// fact rather than a constant — see the bound in the body — and the block
/// count is a set of competing alternatives rather than one pre-picked
/// divisor. Both are stopgaps for `FoldDomain::blocks`; neither is
/// part of the law.
///
/// **Every operand is blocked.** A fold is multi-operand now, so the law's
/// operand clause is not decoration: one blocking view is applied to each
/// input, and inputs that do not agree on the shape it is stated against
/// decline. Reading only `ins[0]` would have silently refused to split every
/// fold the carrier laws mint.
fn fold_split(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L0(L0::Fold {
        carrier,
        axis,
        acc,
        ins,
    }) = &node.op
    else {
        return None;
    };
    // The outer level reads partial *accumulators*, so it may only exist when
    // the carrier is associative; `f.own().numeric` is the meet over every
    // operand, which `f.numeric(0)` is blind to on a multi-operand fold.
    if !carrier.associative || !f.own().numeric.reassoc {
        return None;
    }
    if acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }
    // A rounding lift is the QAT fake-quant value. `infer_l0` does not yet
    // derive `STRICT` from `ScalarKind::Round` (a documented gap in the
    // semantics crate), so the meet above cannot see it once ABSORB has moved
    // the rounding *into* the lift. Reading the carrier directly is what
    // keeps that absorption from laundering a non-reassociable value into a
    // splittable one.
    if carrier.lift.iter().any(has_round) || carrier.merge.iter().any(has_round) {
        return None;
    }
    if ins.is_empty() {
        return None;
    }
    // The law states a TWO-level split, and both levels are already here: a
    // level that is itself a level of a split does not split again. Without
    // this the rewrite cascades — every minted fold is a fresh id, so the
    // driver offers it the rule again — and one reduction over 65,536
    // elements grew the graph to 5,819 nodes. The deleted `at_least(4096)`
    // gate was silently doing this job as well as its own.
    if ins.iter().any(|&x| stands_on_a_split(b, x, 4)) {
        return None;
    }
    let axis = *axis as usize;
    let shape = f.operand(0)?.shape.clone();
    // **Every operand is blocked**, which is what makes the law apply to the
    // multi-operand folds ABSORB and the carrier laws mint rather than only to
    // the single-operand ones the frontend emits. One blocking view serves all
    // of them, so they must agree on the shape it is stated against; an
    // operand at a different shape would need its own view and its own proof
    // that the two still name the same element, so it declines rather than
    // guessing. (`fold_elide` already narrows every operand — this is the same
    // treatment on the other clause.)
    for i in 1..ins.len() {
        let other = &f.operand(i)?.shape;
        if other.len() != shape.len()
            || !other.iter().zip(shape.iter()).all(|(a, c)| a.known_eq(*c))
        {
            return None;
        }
    }
    // `Dim::Sym` declines: `StrideSpec::multiplier` is a `u32`, so the inner
    // extent has to be spellable. That is a limit of `Restride`, not of the
    // law — a symbolic-length reduction gets every other fold law and not
    // this one.
    let extent = shape.get(axis)?.as_const()?;
    // The one bound left, and it is a DEVICE fact rather than a constant: a
    // reduction one workgroup's lanes already cover has nowhere to put a
    // second level, so blocking it buys no parallelism and costs a dispatch.
    // (It costs a dispatch because `fusor2-cost`'s realize step still forces a
    // launch boundary on every fold-to-fold edge, so the nested-loop reading
    // of a split — the one this law is for — is not priceable yet. When that
    // boundary is repaired and `FoldDomain` carries `blocks`, this bound and
    // the candidate enumeration below both go away and the whole thing is a
    // schedule point.) The shipped `at_least(4096)` refused every extent the
    // trainer and the conformance matmul cases use; this fires at 512 and up
    // under the WebGPU baseline limit.
    if extent <= u64::from(f.caps().limits.max_compute_invocations_per_workgroup) {
        return None;
    }

    let mut minted = None;
    for blocks in block_candidates(extent) {
        let inner = extent / blocks;
        let Ok(inner_mult) = u32::try_from(inner) else {
            continue;
        };

        let mut specs: SmallVec<[StrideSpec; 6]> = SmallVec::new();
        for (j, d) in shape.iter().enumerate() {
            if j == axis {
                specs.push(StrideSpec::dim_with(
                    axis as u32,
                    Dim::Const(blocks),
                    inner_mult,
                ));
                specs.push(StrideSpec::dim(axis as u32, Dim::Const(inner)));
            } else {
                specs.push(StrideSpec::dim(j as u32, *d));
            }
        }

        let mut blocked: SmallVec<[Id; 4]> = SmallVec::new();
        for &x in ins {
            let Ok(v) = b.add_l0(L0::Restride {
                specs: specs.clone(),
                bounds: BoundsProof::RuntimeMask,
                x,
            }) else {
                break;
            };
            blocked.push(v);
        }
        if blocked.len() != ins.len() {
            continue;
        }
        let Ok(partial) = b.add_l0(L0::Fold {
            carrier: carrier.clone(),
            axis: axis as u32 + 1,
            acc: *acc,
            ins: blocked,
        }) else {
            continue;
        };
        // **The outer level's elements are partial accumulators, not raw
        // elements**, so it must use `as_merge` — the carrier with `lift`
        // replaced by the identity injection `Arg(k)`, reading the one
        // operand that carries the inner fold's trailing carrier axis.
        let Ok(joined) = b.add_l0(L0::Fold {
            carrier: carrier.as_merge(),
            axis: axis as u32,
            acc: *acc,
            ins: smallvec::smallvec![partial],
        }) else {
            continue;
        };
        // Every candidate joins the class. Building the node without unioning
        // it leaves it unreachable from the root and the choice unmade.
        minted = b.union(id, joined).ok().or(minted);
    }
    minted
}

/// Whether a SPLIT already stands under `x`: a blocked view of one axis, or a
/// fold reading one. The blocking spelling is two adjacent [`StrideSpec`]s
/// naming the same `input_dim`, which is what
/// [`fold_split`] mints and nothing else does.
fn stands_on_a_split(b: &Builder<'_>, x: Id, budget: u32) -> bool {
    if budget == 0 {
        return false;
    }
    match &b.node(x).op {
        Op::L0(L0::Restride { specs, x: inner, .. }) => {
            specs
                .windows(2)
                .any(|w| w[0].input_dim == w[1].input_dim && w[0].multiplier > 1)
                || stands_on_a_split(b, *inner, budget - 1)
        }
        Op::L0(L0::Fold { ins, .. }) => ins
            .iter()
            .any(|&i| stands_on_a_split(b, i, budget - 1)),
        _ => false,
    }
}

/// Candidate block counts for one extent: the power-of-two divisors, widest
/// first, capped at [`MAX_SPLIT_CANDIDATES`].
///
/// This is the piece that belongs in `ScheduleDomain::Fold`. Until
/// `FoldDomain` carries `blocks`, minting the candidates as alternatives is
/// what keeps the factor a *decision* rather than a constant the rule picked:
/// every one of them is priced on the realized DAG, and the unsplit form
/// stays live beside them.
const MAX_SPLIT_CANDIDATES: usize = 3;

fn block_candidates(extent: u64) -> SmallVec<[u64; 4]> {
    [64u64, 32, 16, 8, 4, 2]
        .into_iter()
        .filter(|bl| extent % bl == 0 && extent / bl > 1)
        .take(MAX_SPLIT_CANDIDATES)
        .collect()
}

/// ELIDE: a reduction whose lift is the carrier's identity outside a
/// contiguous range of the reduced axis equals the same reduction over that
/// range alone, because `merge(acc, identity) = acc`.
///
/// This is the entire content of a mask. The frontend compiles padding to
/// `select(IndexOf(a) < valid, x, 0)` and `0` is the `Add` identity; it
/// compiles causality to `select(.., .., -inf)` and `-inf` is the `Max`
/// identity while `exp(-inf) = 0` is the `Add` identity. Both are identity
/// blocks *by construction*, so the compiler skips them by a general law
/// rather than a `causal: bool` on a bespoke node.
///
/// **Decidable, or nothing.** The narrowed range is computed from a predicate
/// affine in `IndexOf(axis)` against a closed bound. A bound that reads a free
/// index — the causal `IndexOf(lk) <= IndexOf(lq) + d` — narrows the domain
/// *per row*, which no `IndexSpace` in this IR can express today; that case
/// declines rather than guessing, and so does anything `eval_closed` cannot
/// decide.
fn fold_elide(b: &mut Builder<'_>, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L0(L0::Fold {
        carrier,
        axis,
        acc,
        ins,
    }) = &node.op
    else {
        return None;
    };
    let axis_u = *axis as usize;
    let shape = f.operand(0)?.shape.clone();
    let extent = shape.get(axis_u)?.as_const()?;

    // Every slot must be guarded by the same predicate and fall back to that
    // slot's own identity: a carrier whose `l` slot lifts to `Lit(1)` is not
    // identity-valued, and eliding it would drop a count.
    let mut cond: Option<ScalarExpr> = None;
    let mut bodies: SmallVec<[ScalarExpr; 4]> = SmallVec::new();
    for (k, l) in carrier.lift.iter().enumerate() {
        let ScalarKind::Select { c, t, f: alt } = l.kind() else {
            return None;
        };
        let rest = eval_closed(alt)?;
        if rest != *carrier.identity.get(k)? {
            return None;
        }
        match &cond {
            Some(prev) if prev != c => return None,
            Some(_) => {}
            None => cond = Some(c.clone()),
        }
        bodies.push(t.clone());
    }
    let (lo, hi) = true_range(cond.as_ref()?, *axis, extent)?;
    // An empty range would make the whole fold the identity — a `Const` leaf,
    // not a narrowing. Nothing in the suite mints one and the honest spelling
    // needs a shape; decline instead of guessing.
    if lo >= hi || (lo == 0 && hi == extent) {
        return None;
    }
    // Narrowing to `[lo, hi)` renumbers the reduced coordinate down by `lo`.
    // A body that names that coordinate — an ALiBi or positional term — would
    // silently read the wrong index, so it declines unless the window starts
    // at zero.
    if lo > 0 && bodies.iter().any(|e| reads_index_of(e, *axis)) {
        return None;
    }

    let specs: SmallVec<[StrideSpec; 6]> = shape
        .iter()
        .enumerate()
        .map(|(j, d)| {
            if j == axis_u {
                StrideSpec::dim(j as u32, Dim::Const(hi - lo)).with_offset(Dim::Const(lo))
            } else {
                StrideSpec::dim(j as u32, *d)
            }
        })
        .collect();
    let mut narrowed: SmallVec<[Id; 4]> = SmallVec::new();
    for &x in ins {
        narrowed.push(
            b.add_l0(L0::Restride {
                specs: specs.clone(),
                bounds: BoundsProof::Static,
                x,
            })
            .ok()?,
        );
    }
    b.add_l0(L0::Fold {
        carrier: carrier.clone().with_lift(bodies),
        axis: *axis,
        acc: *acc,
        ins: narrowed,
    })
    .ok()
}

/// The contiguous range of `axis` on which `cond` is true, or `None` when
/// that is not decidable. Conservative by construction: an undecidable
/// predicate does not narrow.
fn true_range(cond: &ScalarExpr, axis: u32, extent: u64) -> Option<(u64, u64)> {
    let ScalarKind::Cmp { op, a, b } = cond.kind() else {
        return None;
    };
    // One side names the reduced coordinate; the other must be closed, so the
    // bound is the same for every row. `IndexOf` of a *free* axis is the
    // triangular case and is not expressible as a range on this node.
    let (op, bound) = match (a.kind(), b.kind()) {
        (ScalarKind::IndexOf(i), _) if *i == axis => (*op, eval_closed(b)?),
        (_, ScalarKind::IndexOf(i)) if *i == axis => (flip(*op), eval_closed(a)?),
        _ => return None,
    };
    let v = to_f64(bound);
    if !v.is_finite() || v.fract() != 0.0 || v < 0.0 || v > u32::MAX as f64 {
        return None;
    }
    let c = v as u64;
    let clamp = |x: u64| x.min(extent);
    Some(match op {
        CmpOp::Lt => (0, clamp(c)),
        CmpOp::Le => (0, clamp(c.saturating_add(1))),
        CmpOp::Gt => (clamp(c.saturating_add(1)), extent),
        CmpOp::Ge => (clamp(c), extent),
        CmpOp::Eq => (clamp(c), clamp(c.saturating_add(1))),
        // `!=` leaves a hole in the middle: contiguous only at an end.
        CmpOp::Ne => match c {
            0 => (1, extent),
            _ if c + 1 == extent => (0, extent - 1),
            _ => return None,
        },
    })
}

/// `a op b` read as `b op' a`.
fn flip(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::Le => CmpOp::Ge,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::Ge => CmpOp::Le,
        CmpOp::Eq => CmpOp::Eq,
        CmpOp::Ne => CmpOp::Ne,
    }
}

/// Whether `e` names the loop coordinate of `axis`.
fn reads_index_of(e: &ScalarExpr, axis: u32) -> bool {
    match e.kind() {
        ScalarKind::IndexOf(a) => *a == axis,
        ScalarKind::Un { x, .. }
        | ScalarKind::Cast { x, .. }
        | ScalarKind::Bitcast { x, .. }
        | ScalarKind::Round { x, .. }
        | ScalarKind::Splat { x, .. } => reads_index_of(x, axis),
        ScalarKind::Bin { a, b, .. } | ScalarKind::Cmp { a, b, .. } | ScalarKind::Dot { a, b } => {
            reads_index_of(a, axis) || reads_index_of(b, axis)
        }
        ScalarKind::Select { c, t, f } => {
            reads_index_of(c, axis) || reads_index_of(t, axis) || reads_index_of(f, axis)
        }
        ScalarKind::Arg(_) | ScalarKind::Lit(_) | ScalarKind::Uniform(_) => false,
    }
}

/// Whether `e` rounds anywhere: the one syntactic marker of a value whose
/// contract forbids reassociation.
fn has_round(e: &ScalarExpr) -> bool {
    match e.kind() {
        ScalarKind::Round { .. } => true,
        ScalarKind::Un { x, .. }
        | ScalarKind::Cast { x, .. }
        | ScalarKind::Bitcast { x, .. }
        | ScalarKind::Splat { x, .. } => has_round(x),
        ScalarKind::Bin { a, b, .. } | ScalarKind::Cmp { a, b, .. } | ScalarKind::Dot { a, b } => {
            has_round(a) || has_round(b)
        }
        ScalarKind::Select { c, t, f } => has_round(c) || has_round(t) || has_round(f),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Additive contraction recognition
// ---------------------------------------------------------------------------

/// `Fold{Add, rank-1}(Map{mul(Arg0, Arg1)}(a, b))` also *is* a `Contract`.
///
/// There is no sole-reader gate, no cached check and no `commit_recognized`:
/// the `mul`+`fold` form stays live in the same class, so a product read
/// twice keeps both options open and the extractor prices them.
pub fn recognize_contract(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L0(L0::Fold {
        carrier,
        axis,
        acc,
        ins: fold_ins,
    }) = &node.op
    else {
        return None;
    };
    if carrier.kind() != Some(BinOp::Add) || carrier.slots.len() != 1 {
        return None;
    }
    // **Both spellings of "multiply, then sum".**
    //
    // The product may sit in a separate `Map` the fold reads, or directly in
    // the carrier's own `lift` — `Carrier::binop(Add).with_lift(mul(Arg0,
    // Arg1))` is exactly what `lower_generic` mints and what `ABSORB` leaves
    // behind once it inlines the map into the lift. Matching only the first
    // spelling made this rule unreachable on any chain fusion had already
    // touched, which is every interesting one: attention's score matmul
    // arrives here as the second form and so was never recognized as a
    // contraction at all.
    let ins: SmallVec<[Id; 2]> = if carrier.lift[0].kind() == &ScalarKind::Arg(0) {
        let &[x] = &fold_ins[..] else {
            return None;
        };
        let Op::L0(L0::Map { expr, ins, outs }) = b.node(x).op.clone() else {
            return None;
        };
        if outs != 1 || ins.len() != 2 || !is_arg_product(&expr) {
            return None;
        }
        smallvec::smallvec![ins[0], ins[1]]
    } else if is_arg_product(&carrier.lift[0]) {
        let &[p, q] = &fold_ins[..] else {
            return None;
        };
        smallvec::smallvec![p, q]
    } else {
        return None;
    };
    let rank = f.operand(0)?.shape.len();
    if rank == 0 || rank > u8::MAX as usize || *axis as usize != rank - 1 {
        return None;
    }
    // **Read each operand through its broadcast, not around it.**
    //
    // Stating both sides with all `rank` labels only describes the aligned
    // case, where `a` and `b` genuinely hold every axis. The moment either
    // side reaches the product through a broadcast — which is how *every*
    // outer-product-shaped contraction is spelled, `Restride` with
    // `multiplier == 0` — that spec names a contraction over the materialized
    // broadcast. For attention's scores that is two `[B,H,Lq,Lk,Dh]` operands
    // where `[B,H,Lq,Dh]` and `[B,H,Lk,Dh]` were meant: 2 GB of operand where
    // there should be 2 MB, and the extractor rightly declines it, so the fold
    // stays a rank-5 generic reduce and never reaches `Family::Coop`.
    //
    // Naming only the axes an operand varies along makes the spec the real
    // einsum — `bhqd,bhkd->bhqk` — and every contraction lowering, coop
    // included, becomes available. Measured on `[1,8,1024,64]` attention: the
    // score matmul goes from a `KFold` over `[1,8,1024,1024,64]` to a
    // `KContract{Coop}`, which is the whole of the gap against the reference's
    // hand-written flash kernel on that half of the chain.
    let (a_src, a_labels) = contract_operand(b, ins[0], rank)?;
    let (b_src, b_labels) = contract_operand(b, ins[1], rank)?;
    let contracted = Label(rank as u8 - 1);
    // The reduced axis has to be a real shared axis of both operands. Where
    // one side is broadcast along it the fold is a scaled sum, not a
    // contraction, and `verify_l0`'s "every label in >= 2 of {a,b,out}" would
    // reject the spec anyway.
    if !a_labels.contains(&contracted) || !b_labels.contains(&contracted) {
        return None;
    }
    let out: SmallVec<[Label; 6]> = (0..rank as u8 - 1)
        .map(Label)
        .filter(|l| a_labels.contains(l) || b_labels.contains(l))
        .collect();
    let spec = EinSpec {
        a: a_labels,
        b: b_labels,
        out,
    };
    let contracted = b
        .add_l0(L0::Contract {
            spec,
            acc: *acc,
            a: a_src,
            b: b_src,
            outs: 1,
        })
        .ok()?;
    b.union(id, contracted).ok()
}

/// The value a contraction should read for this operand, and the labels it
/// actually varies along.
///
/// An operand that reaches the product through a single `Restride` which only
/// *broadcasts* — every kept axis read densely in order, every dropped axis
/// `multiplier == 0` — is really its base read at the base's own labels. Any
/// other view (a permute, a narrowing offset, a strided or multi-node spine)
/// is left alone: this returns the operand as given with all `rank` labels,
/// which is the shipped behaviour.
fn contract_operand(
    b: &Builder<'_>,
    v: Id,
    rank: usize,
) -> Option<(Id, SmallVec<[Label; 6]>)> {
    let all = || -> SmallVec<[Label; 6]> { (0..rank as u8).map(Label).collect() };
    let spine = b.trace_pure_views(v);
    if spine.views.len() != 1 {
        return Some((v, all()));
    }
    let Op::L0(L0::Restride { specs, .. }) = b.node(spine.views[0]).op.clone() else {
        return Some((v, all()));
    };
    if specs.len() != rank {
        return Some((v, all()));
    }
    let base_shape = b.facts_of(spine.base).shape.clone();
    let mut labels: SmallVec<[Label; 6]> = SmallVec::new();
    let mut next_base = 0usize;
    for (i, s) in specs.iter().enumerate() {
        if s.multiplier == 0 {
            // A broadcast axis. It is not one of this operand's labels, and
            // `Contract` will re-broadcast it from the spec.
            continue;
        }
        // Every axis this operand does vary along has to be the next axis of
        // the base, read whole and in order. Anything else is a view this
        // rewrite cannot restate as a label.
        let dim = *base_shape.get(next_base)?;
        if s.input_dim as usize != next_base
            || s.multiplier != 1
            || !s.offset.known_eq(Dim::Const(0))
            || !s.size.known_eq(dim)
        {
            return Some((v, all()));
        }
        labels.push(Label(i as u8));
        next_base += 1;
    }
    // Every base axis has to be accounted for, or the view is doing something
    // other than inserting broadcasts.
    if next_base != base_shape.len() {
        return Some((v, all()));
    }
    if labels.len() == rank {
        // Nothing was broadcast; this is the shipped aligned case.
        return Some((v, all()));
    }
    Some((spine.base, labels))
}

fn is_arg_product(e: &ScalarExpr) -> bool {
    let ScalarKind::Bin {
        op: BinOp::Mul,
        a,
        b,
    } = e.kind()
    else {
        return false;
    };
    matches!((a.kind(), b.kind()), (ScalarKind::Arg(0), ScalarKind::Arg(1)))
        || matches!((a.kind(), b.kind()), (ScalarKind::Arg(1), ScalarKind::Arg(0)))
}

/// `Contract(Contract(a, b), c)` also equals `Contract(a, Contract(b, c))`.
///
/// Legal when the two specs share one consistent labelling and neither
/// regrouping captures a label: an inner-summed label may not reappear in
/// `c`, and an outer-summed label may not appear in `a`. Every operand must
/// permit reassociation.
pub fn contract_reassoc(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L0(L0::Contract {
        spec: outer,
        acc,
        a: inner_id,
        b: c_id,
        outs: 1,
    }) = &node.op
    else {
        return None;
    };
    if !f.operands().iter().all(|o| o.numeric.reassoc) {
        return None;
    }
    let Op::L0(L0::Contract {
        spec: inner,
        acc: inner_acc,
        a: a_id,
        b: b_id,
        outs: 1,
    }) = b.node(*inner_id).op.clone()
    else {
        return None;
    };

    let (la, lb, lt) = (&inner.a, &inner.b, &inner.out);
    let (lt2, lc, lo) = (&outer.a, &outer.b, &outer.out);
    // One consistent labelling: the inner result enters the outer under the
    // same names it left under.
    if lt != lt2 {
        return None;
    }
    let has = |v: &SmallVec<[Label; 6]>, l: &Label| v.contains(l);
    let k1: Vec<Label> = la
        .iter()
        .copied()
        .filter(|l| has(lb, l) && !has(lo, l))
        .collect();
    let k2: Vec<Label> = lt
        .iter()
        .copied()
        .filter(|l| has(lc, l) && !has(lo, l))
        .collect();
    if k1.iter().any(|l| has(lc, l)) || k2.iter().any(|l| has(la, l)) {
        return None;
    }
    if la.iter().any(|l| has(lc, l) && !has(lo, l)) {
        return None;
    }

    // The regrouped intermediate keeps exactly the labels something
    // downstream still needs.
    let mut lu: SmallVec<[Label; 6]> = SmallVec::new();
    for l in lb.iter().chain(lc.iter()) {
        if (has(la, l) || has(lo, l)) && !lu.contains(l) {
            lu.push(*l);
        }
    }
    lu.sort_unstable();

    let regrouped = b
        .add_l0(L0::Contract {
            spec: EinSpec {
                a: lb.clone(),
                b: lc.clone(),
                out: lu.clone(),
            },
            acc: inner_acc,
            a: b_id,
            b: *c_id,
            outs: 1,
        })
        .ok()?;
    let joined = b
        .add_l0(L0::Contract {
            spec: EinSpec {
                a: la.clone(),
                b: lu,
                out: lo.clone(),
            },
            acc: *acc,
            a: a_id,
            b: regrouped,
            outs: 1,
        })
        .ok()?;
    b.union(id, joined).ok()
}

// ---------------------------------------------------------------------------
// Closed-expression folding and identity elimination
// ---------------------------------------------------------------------------

/// A `Map` whose body is closed over literals alone also equals a constant
/// leaf. Additive: the unfolded form survives, so a target that would rather
/// recompute than read a splat still can.
pub fn const_fold_map(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L0(L0::Map { expr, outs: 1, .. }) = &node.op else {
        return None;
    };
    let value = eval_closed(expr)?;
    let folded = b
        .add_l0(L0::Leaf(LeafKind::Const {
            value,
            shape: f.own().shape.clone(),
        }))
        .ok()?;
    b.union(id, folded).ok()
}

/// The closed interpreter `const_fold_map` runs. Declines on `Arg`,
/// `Uniform`, `IndexOf`, `Dot` and `Splat`, which is what makes it closed.
fn eval_closed(e: &ScalarExpr) -> Option<Splat> {
    let out = e.dtype();
    match e.kind() {
        ScalarKind::Lit(Lit(v)) => Some(*v),
        ScalarKind::Un { op, x } => {
            let v = to_f64(eval_closed(x)?);
            from_f64(apply_un(*op, v)?, out)
        }
        ScalarKind::Bin { op, a, b } => {
            let (x, y) = (to_f64(eval_closed(a)?), to_f64(eval_closed(b)?));
            from_f64(apply_bin(*op, x, y, out)?, out)
        }
        ScalarKind::Cmp { op, a, b } => {
            let (x, y) = (to_f64(eval_closed(a)?), to_f64(eval_closed(b)?));
            let t = match op {
                CmpOp::Lt => x < y,
                CmpOp::Le => x <= y,
                CmpOp::Gt => x > y,
                CmpOp::Ge => x >= y,
                CmpOp::Eq => x == y,
                CmpOp::Ne => x != y,
            };
            from_f64(if t { 1.0 } else { 0.0 }, out)
        }
        ScalarKind::Select { c, t, f } => {
            if to_f64(eval_closed(c)?) != 0.0 {
                eval_closed(t)
            } else {
                eval_closed(f)
            }
        }
        ScalarKind::Cast { to, x } => from_f64(to_f64(eval_closed(x)?), *to),
        ScalarKind::Bitcast { to, x } => from_bits(eval_closed(x)?.bits(), *to),
        ScalarKind::Round { mode, x } => {
            let v = to_f64(eval_closed(x)?);
            from_f64(apply_round(*mode, v), out)
        }
        ScalarKind::Arg(_)
        | ScalarKind::Uniform(_)
        | ScalarKind::IndexOf(_)
        | ScalarKind::Dot { .. }
        | ScalarKind::Splat { .. } => None,
    }
}

fn to_f64(s: Splat) -> f64 {
    match s {
        Splat::F32(v) => f64::from(v),
        Splat::F16(bits) => half::f16::from_bits(bits).to_f64(),
        Splat::BF16(bits) => half::bf16::from_bits(bits).to_f64(),
        Splat::U32(v) => f64::from(v),
        Splat::I32(v) => f64::from(v),
    }
}

fn from_f64(v: f64, d: Dtype) -> Option<Splat> {
    Some(match d {
        Dtype::F32 => Splat::F32(v as f32),
        Dtype::F16 => Splat::F16(half::f16::from_f64(v).to_bits()),
        Dtype::BF16 => Splat::BF16(half::bf16::from_f64(v).to_bits()),
        Dtype::U32 => Splat::U32(v as u32),
        Dtype::I32 => Splat::I32(v as i32),
        Dtype::Q(_) => return None,
    })
}

fn from_bits(bits: u32, d: Dtype) -> Option<Splat> {
    Some(match d {
        Dtype::F32 => Splat::F32(f32::from_bits(bits)),
        Dtype::F16 => Splat::F16(bits as u16),
        Dtype::BF16 => Splat::BF16(bits as u16),
        Dtype::U32 => Splat::U32(bits),
        Dtype::I32 => Splat::I32(bits as i32),
        Dtype::Q(_) => return None,
    })
}

fn apply_un(op: UnOp, v: f64) -> Option<f64> {
    Some(match op {
        // A relaxed accuracy contract does not license a *different* constant.
        UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp => v.exp(),
        UnOp::Exp2 => v.exp2(),
        UnOp::Log => v.ln(),
        UnOp::Log2 => v.log2(),
        UnOp::Sqrt => v.sqrt(),
        UnOp::InverseSqrt => 1.0 / v.sqrt(),
        UnOp::Sin => v.sin(),
        UnOp::Cos => v.cos(),
        UnOp::Tan => v.tan(),
        UnOp::Tanh => v.tanh(),
        UnOp::Asin => v.asin(),
        UnOp::Acos => v.acos(),
        UnOp::Atan => v.atan(),
        UnOp::Sinh => v.sinh(),
        UnOp::Cosh => v.cosh(),
        UnOp::Asinh => v.asinh(),
        UnOp::Acosh => v.acosh(),
        UnOp::Atanh => v.atanh(),
        UnOp::Abs => v.abs(),
        UnOp::Neg => -v,
        // Width-sensitive bit surgery: not a closed scalar identity.
        UnOp::Unpack2x16Float => return None,
    })
}

fn apply_bin(op: BinOp, x: f64, y: f64, d: Dtype) -> Option<f64> {
    let integral = d.is_int();
    Some(match op {
        BinOp::Add => x + y,
        BinOp::Sub => x - y,
        BinOp::Mul => x * y,
        BinOp::Div => {
            if integral {
                if y == 0.0 {
                    return None;
                }
                (x / y).trunc()
            } else {
                x / y
            }
        }
        BinOp::Rem => {
            if y == 0.0 {
                return None;
            }
            x % y
        }
        BinOp::Pow => x.powf(y),
        BinOp::Min => x.min(y),
        BinOp::Max => x.max(y),
        BinOp::BitAnd => int_op(x, y, |a, b| a & b)?,
        BinOp::BitOr => int_op(x, y, |a, b| a | b)?,
        BinOp::BitXor => int_op(x, y, |a, b| a ^ b)?,
        BinOp::Shr => int_op(x, y, |a, b| a >> (b & 31))?,
        BinOp::Shl => int_op(x, y, |a, b| a << (b & 31))?,
        BinOp::LogicalAnd => {
            if x != 0.0 && y != 0.0 {
                1.0
            } else {
                0.0
            }
        }
        BinOp::LogicalOr => {
            if x != 0.0 || y != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    })
}

fn int_op(x: f64, y: f64, f: impl Fn(i64, i64) -> i64) -> Option<f64> {
    if x.fract() != 0.0 || y.fract() != 0.0 {
        return None;
    }
    Some(f(x as i64, y as i64) as f64)
}

fn apply_round(mode: RoundMode, v: f64) -> f64 {
    match mode {
        RoundMode::HalfToEven => {
            let r = v.round();
            if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
                r - v.signum()
            } else {
                r
            }
        }
        RoundMode::HalfAwayFromZero => v.round(),
        RoundMode::Floor => v.floor(),
        RoundMode::Ceil => v.ceil(),
        RoundMode::Trunc => v.trunc(),
    }
}

/// `x+0`, `x-0`, `x*1`, `x/1`, `pow(x,1)`, `select(lit, t, f)` and a cast to
/// the type the value already has are all identities; so is a `Restride`
/// standing between this map and its producer whose spec vector is the
/// identity view. Rewriting the body in place and dropping the pass-through
/// view are one rule, minted at the reading node.
pub fn identity_elim(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::L0(L0::Map { expr, ins, outs }) = &node.op else {
        return None;
    };
    let (body, body_changed) = simplify(expr);

    let mut new_ins = ins.clone();
    let mut ins_changed = false;
    for slot in new_ins.iter_mut() {
        let Op::L0(L0::Restride { specs, x, .. }) = b.node(*slot).op.clone() else {
            continue;
        };
        if crate::rules::is_identity_specs(&specs, &b.facts_of(x).shape) {
            *slot = x;
            ins_changed = true;
        }
    }
    if !body_changed && !ins_changed {
        return None;
    }
    let simplified = b
        .add_l0(L0::Map {
            expr: body,
            ins: new_ins,
            outs: *outs,
        })
        .ok()?;
    b.union(id, simplified).ok()
}

fn simplify(e: &ScalarExpr) -> (ScalarExpr, bool) {
    let rebuilt = match e.kind() {
        ScalarKind::Un { op, x } => {
            let (x, c) = simplify(x);
            (ScalarExpr::un(*op, x), c)
        }
        ScalarKind::Bin { op, a, b } => {
            let (a, ca) = simplify(a);
            let (bb, cb) = simplify(b);
            (ScalarExpr::bin(*op, a, bb), ca || cb)
        }
        ScalarKind::Cmp { op, a, b } => {
            let (a, ca) = simplify(a);
            let (bb, cb) = simplify(b);
            (ScalarExpr::cmp(*op, a, bb), ca || cb)
        }
        ScalarKind::Select { c, t, f } => {
            let (c, cc) = simplify(c);
            let (t, ct) = simplify(t);
            let (f, cf) = simplify(f);
            (ScalarExpr::select(c, t, f), cc || ct || cf)
        }
        ScalarKind::Cast { to, x } => {
            let (x, c) = simplify(x);
            (ScalarExpr::cast(*to, x), c)
        }
        ScalarKind::Bitcast { to, x } => {
            let (x, c) = simplify(x);
            (ScalarExpr::bitcast(*to, x), c)
        }
        ScalarKind::Round { mode, x } => {
            let (x, c) = simplify(x);
            (ScalarExpr::round(*mode, x), c)
        }
        _ => (e.clone(), false),
    };
    let (node, changed) = rebuilt;
    match peephole(&node) {
        Some(simpler) => (simpler, true),
        None => (node, changed),
    }
}

fn peephole(e: &ScalarExpr) -> Option<ScalarExpr> {
    match e.kind() {
        ScalarKind::Bin { op, a, b } => match op {
            BinOp::Add => {
                if lit_is(b, 0.0) {
                    Some(a.clone())
                } else if lit_is(a, 0.0) {
                    Some(b.clone())
                } else {
                    None
                }
            }
            BinOp::Sub => lit_is(b, 0.0).then(|| a.clone()),
            BinOp::Mul => {
                if lit_is(b, 1.0) {
                    Some(a.clone())
                } else if lit_is(a, 1.0) {
                    Some(b.clone())
                } else {
                    None
                }
            }
            BinOp::Div | BinOp::Pow => lit_is(b, 1.0).then(|| a.clone()),
            _ => None,
        },
        ScalarKind::Select { c, t, f } => {
            let ScalarKind::Lit(Lit(v)) = c.kind() else {
                return None;
            };
            Some(if to_f64(*v) != 0.0 { t.clone() } else { f.clone() })
        }
        ScalarKind::Cast { to, x } => (*to == x.dtype()).then(|| x.clone()),
        _ => None,
    }
}

fn lit_is(e: &ScalarExpr, v: f64) -> bool {
    matches!(e.kind(), ScalarKind::Lit(Lit(s)) if to_f64(*s) == v)
}

/// The type side of the `widen-compute` lowering rule: a `Map` storing F16 or
/// BF16 also equals the same arithmetic performed at
/// [`Dtype::compute_dtype`] with a trailing narrowing cast. The CPU target
/// contributes the lane-level counterpart.
pub fn widen_store_cast(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L0(L0::Map { expr, ins, outs }) = &node.op else {
        return None;
    };
    let narrow = expr.dtype();
    if !matches!(narrow, Dtype::F16 | Dtype::BF16) {
        return None;
    }
    if f.own().numeric.min_accum_bits > 32 {
        return None;
    }
    // Already in widened form; re-firing would only re-cast.
    if matches!(expr.kind(), ScalarKind::Cast { .. }) {
        return None;
    }
    let widened = ScalarExpr::cast(narrow, widen(expr)?);
    let alt = b
        .add_l0(L0::Map {
            expr: widened,
            ins: ins.clone(),
            outs: *outs,
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// Rebuild `e` at [`Dtype::compute_dtype`]. Declines on `Bitcast`, whose
/// meaning depends on the storage width.
fn widen(e: &ScalarExpr) -> Option<ScalarExpr> {
    let up = |x: &ScalarExpr| -> Option<ScalarExpr> { widen(x) };
    Some(match e.kind() {
        ScalarKind::Arg(i) => {
            let d = e.dtype();
            let a = ScalarExpr::arg(*i, d);
            if d.compute_dtype() == d {
                a
            } else {
                ScalarExpr::cast(d.compute_dtype(), a)
            }
        }
        ScalarKind::Uniform(s) => {
            let d = e.dtype();
            let u = ScalarExpr::uniform(*s, d);
            if d.compute_dtype() == d {
                u
            } else {
                ScalarExpr::cast(d.compute_dtype(), u)
            }
        }
        ScalarKind::Lit(Lit(v)) => {
            let d = v.dtype();
            if d.compute_dtype() == d {
                e.clone()
            } else {
                ScalarExpr::lit(from_f64(to_f64(*v), d.compute_dtype())?)
            }
        }
        ScalarKind::IndexOf(a) => ScalarExpr::index_of(*a),
        ScalarKind::Un { op, x } => ScalarExpr::un(*op, up(x)?),
        ScalarKind::Bin { op, a, b } => ScalarExpr::bin(*op, up(a)?, up(b)?),
        ScalarKind::Cmp { op, a, b } => ScalarExpr::cmp(*op, up(a)?, up(b)?),
        ScalarKind::Select { c, t, f } => ScalarExpr::select(up(c)?, up(t)?, up(f)?),
        ScalarKind::Cast { to, x } => ScalarExpr::cast(to.compute_dtype(), up(x)?),
        ScalarKind::Round { mode, x } => ScalarExpr::round(*mode, up(x)?),
        ScalarKind::Bitcast { .. } | ScalarKind::Dot { .. } | ScalarKind::Splat { .. } => {
            return None;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::NumericContract;
    use crate::egraph::{SaturationBudget, Saturate};
    use crate::rules::CORE_RULES;
    use crate::rules::test_support as ts;
    use crate::saturate::CoreSaturate;

    fn fold_at(extent: u64, carrier: Carrier) -> (crate::egraph::EGraph, Id) {
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4), Dim::Const(extent)]);
        let fid = ts::fold(&mut g, carrier, 1, Dtype::F32, x);
        (g, fid)
    }

    fn add() -> Carrier {
        ts::binop_carrier(BinOp::Add, Dtype::F32)
    }

    /// The three-slot accumulator a shifted, weighted reduction carries:
    /// `m` a running reference, `l` the shifted sum, `o` the shifted weighted
    /// sum. `o`'s merge is character-for-character `l`'s, because `T` is
    /// applied once per slot of one module element — which is exactly what
    /// makes a block split of it a block loop with a per-block rescale, with
    /// no code anywhere that knows what the slots mean.
    ///
    /// Hand-built as a **fixture**. No rule mints it yet; this is the test
    /// that lands before one does.
    fn mlo(dtype: Dtype) -> Carrier {
        use crate::carrier::SlotTy;
        let e = Carrier::binop_identity(BinOp::Add, dtype).unwrap();
        let arg = |i: u32| ScalarExpr::arg(i, dtype);
        let (m_a, m_b) = (arg(0), arg(3));
        let m = ScalarExpr::bin(BinOp::Max, m_a.clone(), m_b.clone());
        let rescale = |side: ScalarExpr, value: ScalarExpr| {
            ScalarExpr::bin(
                BinOp::Mul,
                value,
                ScalarExpr::un(UnOp::Exp, Carrier::safe_delta(side, m.clone(), e)),
            )
        };
        let both = |value_a: ScalarExpr, value_b: ScalarExpr| {
            ScalarExpr::bin(
                BinOp::Add,
                rescale(m_a.clone(), value_a),
                rescale(m_b.clone(), value_b),
            )
        };
        Carrier {
            slots: smallvec::smallvec![SlotTy::Scalar, SlotTy::Scalar, SlotTy::Scalar],
            identity: smallvec::smallvec![
                Carrier::binop_identity(BinOp::Max, dtype).unwrap(),
                e,
                e
            ],
            lift: smallvec::smallvec![
                arg(0),
                ScalarExpr::lit(Splat::F32(1.0)),
                arg(1)
            ],
            merge: smallvec::smallvec![
                m.clone(),
                both(arg(1), arg(4)),
                both(arg(2), arg(5))
            ],
            associative: true,
            tie: None,
        }
    }

    /// One sequential pass over `rows`, absorbing element by element.
    fn sequential(c: &Carrier, rows: &[[f32; 2]]) -> Vec<f32> {
        rows.iter()
            .fold(c.identity_f32(), |acc, r| c.absorb(&acc, r).unwrap())
    }

    fn fire(g: &mut crate::egraph::EGraph, id: Id, r: &crate::egraph::Rule) -> Option<Id> {
        let caps = ts::caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (r.apply)(&mut b, id, &node, &facts)
    }

    /// Test 1. Split-K, online softmax and Welford are one rule: only the
    /// carrier differs and the rule never reads what it *means*.
    ///
    /// The multi-slot cases are built by `Carrier::tuple` and by the oracle's
    /// own shift-stabilized carrier, so this exercises the same shapes the
    /// deleted `Combine::OnlineSoftmax` / `Combine::Welford` named.
    #[test]
    fn fold_split_fires_on_one_two_and_three_slot_carriers() {
        let pair = add()
            .tuple(&ts::binop_carrier(BinOp::Max, Dtype::F32), &crate::carrier::ArgRemap::identity(1))
            .carrier;
        let triple = pair
            .tuple(&ts::binop_carrier(BinOp::Mul, Dtype::F32), &crate::carrier::ArgRemap::identity(1))
            .carrier;
        for carrier in [add(), pair, triple] {
            let width = carrier.width();
            let (mut g, fid) = fold_at(8192, carrier.clone());
            let before = g.chain(fid).len();
            assert!(fire(&mut g, fid, &STRIP).is_some(), "{width} slots");
            let members = g.chain(fid);
            // One alternative per candidate block count, all in one class.
            assert_eq!(members.len(), before + MAX_SPLIT_CANDIDATES, "{width} slots");
            let outers: Vec<Id> = members.iter().copied().filter(|&m| m != fid).collect();
            assert_eq!(outers.len(), MAX_SPLIT_CANDIDATES);
            for alt in outers {
                let Op::L0(L0::Fold {
                    carrier: outer,
                    ins,
                    ..
                }) = &g.node(alt).op
                else {
                    panic!("expected a Fold alternative");
                };
                // **The outer level reads partial accumulators.** Its lift is
                // the identity injection, never the inner lift; reusing the
                // inner carrier would apply `lift` to a partial max.
                assert_eq!(*outer, carrier.as_merge(), "{width} slots");
                assert_eq!(outer.merge, carrier.merge);
                for (k, l) in outer.lift.iter().enumerate() {
                    assert_eq!(l.kind(), &ScalarKind::Arg(k as u32));
                }
                // Fix 2: ONE operand, carrying the inner fold's trailing
                // carrier axis. `width` operands would overflow
                // `SmallVec<[Id; 4]>` at a `Vector(64)` slot and make the
                // operand list a function of the head dimension.
                assert_eq!(ins.len(), 1, "{width} slots");
                let expect = 2 + usize::from(width > 1);
                assert_eq!(g.facts(ins[0]).shape.len(), expect, "{width} slots");
            }
        }
    }

    /// Test 2. Without `reassoc` the split form is not value-equal.
    #[test]
    fn fold_split_refuses_non_reassociable() {
        let mut g = ts::graph();
        let raw = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4), Dim::Const(8192)]);
        // A `Round` in front makes the folded value non-reassociable — the
        // QAT fake-quant path, verbatim.
        let strict = ts::map(
            &mut g,
            ScalarExpr::round(RoundMode::HalfAwayFromZero, ScalarExpr::arg(0, Dtype::F32)),
            &[raw],
        );
        assert_eq!(g.facts(strict).numeric, NumericContract::STRICT);
        let fid = ts::fold(&mut g, add(), 1, Dtype::F32, strict);
        assert!(fire(&mut g, fid, &STRIP).is_none());
        assert_eq!(g.chain(fid).len(), 1);
    }

    /// STRIP fix 3. **There is no extent gate**, so the law fires at every
    /// reduction length the trainer and the conformance matmul and attention
    /// cases actually use. The shipped rule refused anything under 4096: at
    /// `k = 512..2048` it never fired, so there was no split-K and no KV block
    /// loop at any real shape, and every derivation that assumed one was
    /// admissible on paper and unschedulable in fact.
    ///
    /// A `Dim::Sym` extent still declines — `StrideSpec::multiplier` is a
    /// `u32` and a symbolic inner extent cannot be spelled — and that is a
    /// limit of `Restride`, not of the law.
    #[test]
    fn strip_splits_at_every_real_extent_and_declines_on_sym() {
        for k in [512u64, 768, 1024, 2048] {
            let (mut g, fid) = fold_at(k, add());
            assert!(fire(&mut g, fid, &STRIP).is_some(), "k = {k}");
            let split: Vec<Id> = g
                .chain(fid)
                .into_iter()
                .filter(|&m| m != fid && matches!(g.node(m).op, Op::L0(L0::Fold { .. })))
                .collect();
            assert!(!split.is_empty(), "k = {k}: no split alternative");
        }

        // One named shape, with the block count asserted. 768 = 2^8 * 3, so
        // the power-of-two divisors offered are 64, 32 and 16 — nothing else
        // divides it — and the inner extents are 12, 24 and 48.
        let (mut g, fid) = fold_at(768, add());
        fire(&mut g, fid, &STRIP).unwrap();
        let mut counts: Vec<u64> = g
            .chain(fid)
            .into_iter()
            .filter_map(|m| match &g.node(m).op {
                Op::L0(L0::Fold { axis: 1, ins, .. }) if m != fid => {
                    Some(g.facts(ins[0]).shape[1].as_const()?)
                }
                _ => None,
            })
            .collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![16, 32, 64]);

        let mut g = ts::graph();
        let s = g.fresh_sym();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4), Dim::Sym(s)]);
        let fid = ts::fold(&mut g, add(), 1, Dtype::F32, x);
        assert!(fire(&mut g, fid, &STRIP).is_none());
        assert_eq!(g.chain(fid).len(), 1);
    }

    /// The one bound left on SPLIT is a **device fact**, not a constant: a
    /// reduction one workgroup's lanes already cover has nowhere to put a
    /// second level. It sits exactly at
    /// `Limits::max_compute_invocations_per_workgroup`, so the same law fires
    /// at a different length on a different device — which the deleted
    /// `at_least(4096)` could not do.
    #[test]
    fn split_declines_what_one_workgroup_already_covers() {
        let lanes = u64::from(ts::caps().limits.max_compute_invocations_per_workgroup);
        for (extent, want) in [(lanes / 2, false), (lanes, false), (lanes * 2, true)] {
            let (mut g, fid) = fold_at(extent, add());
            assert_eq!(
                fire(&mut g, fid, &STRIP).is_some(),
                want,
                "extent {extent} against {lanes} lanes"
            );
        }
    }

    /// A fold is **multi-operand**, and SPLIT blocks every one of its inputs.
    ///
    /// The fixture is a two-operand `Fold{Add, lift = Arg(0) * Arg(1)}` — a
    /// dot product, i.e. split-K stated without a `Contract` anywhere — at an
    /// extent the device bound admits. Reading only `ins[0]` would decline
    /// here, which would silently refuse to split every fold ABSORB and the
    /// carrier laws mint, since those are exactly the multi-operand ones.
    ///
    /// Three claims: the rewrite fires on a **saturated** graph; both operands
    /// carry the blocking view (not just the first); and the two-level result
    /// equals one sequential pass, computed here.
    #[test]
    fn split_blocks_every_operand_of_a_multi_operand_fold() {
        let n = 2048u64;
        let mut g = ts::graph();
        let shape = [Dim::Const(2), Dim::Const(n)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let y = ts::buffer(&mut g, Dtype::F32, &shape);
        let product = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::arg(1, Dtype::F32),
        );
        let carrier = add().with_lift([product]);
        let fid = g
            .add(Op::L0(L0::Fold {
                carrier: carrier.clone(),
                axis: 1,
                acc: Dtype::F32,
                ins: smallvec::smallvec![x, y],
            }))
            .unwrap();
        g.add_root(fid);
        let report = CoreSaturate
            .saturate(&mut g, &ts::caps(), CORE_RULES, SaturationBudget::default())
            .unwrap();
        assert!(report.saturated, "{report:?}");
        assert!(
            report.fired.iter().any(|(r, c)| *r == "STRIP" && *c > 0),
            "STRIP did not fire on a two-operand fold: {:?}",
            report.fired
        );

        // The two-level form, with BOTH operands blocked. The outer level
        // reads one operand carrying the inner fold's trailing carrier axis.
        let outer = carrier.as_merge();
        let split = g
            .chain(fid)
            .into_iter()
            .find_map(|m| match &g.node(m).op {
                Op::L0(L0::Fold { carrier: c, axis: 1, ins, .. })
                    if m != fid && *c == outer && ins.len() == 1 =>
                {
                    match &g.node(ins[0]).op {
                        Op::L0(L0::Fold { axis: 2, ins: inner, .. }) => Some(inner.clone()),
                        _ => None,
                    }
                }
                _ => None,
            })
            .expect("a two-level split whose outer level is the carrier's own merge");
        assert_eq!(split.len(), 2, "the inner level dropped an operand");
        for (i, &v) in split.iter().enumerate() {
            let Op::L0(L0::Restride { specs, x: src, .. }) = &g.node(v).op else {
                panic!("operand {i} was not blocked: {:?}", g.node(v).op)
            };
            assert_eq!(*src, if i == 0 { x } else { y }, "operand {i} was rebound");
            // The blocking spelling: two adjacent specs naming the reduced axis.
            assert_eq!(specs.len(), 3);
            assert_eq!(specs[1].input_dim, 1);
            assert_eq!(specs[2].input_dim, 1);
            assert!(specs[1].multiplier > 1);
        }

        // Numerically: the split equals one sequential pass over the pairs.
        let rows: Vec<[f32; 2]> = (0..24)
            .map(|i| [(i as f32) * 0.25 - 3.0, ((i % 7) as f32) * 0.5 - 1.5])
            .collect();
        let one_pass = sequential(&carrier, &rows);
        for blocks in [2usize, 3, 4, 6] {
            let per = rows.len() / blocks;
            let partials: Vec<Vec<f32>> =
                rows.chunks(per).map(|c| sequential(&carrier, c)).collect();
            let joined = partials
                .iter()
                .fold(outer.identity_f32(), |acc, p| outer.absorb(&acc, p).unwrap());
            assert!(
                (joined[0] - one_pass[0]).abs() <= 1e-4 * one_pass[0].abs().max(1.0),
                "{blocks} blocks: {} vs {}",
                joined[0],
                one_pass[0]
            );
        }
        // Independently: the value is the dot product of the two rows.
        let want: f32 = rows.iter().map(|r| r[0] * r[1]).sum();
        assert!((one_pass[0] - want).abs() <= 1e-4 * want.abs().max(1.0));
    }

    /// One blocking view serves every operand, so operands that disagree on
    /// the shape it is stated against decline rather than being blocked
    /// through a view that does not describe them.
    #[test]
    fn split_declines_operands_at_disagreeing_shapes() {
        let n = 2048u64;
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(2), Dim::Const(n)]);
        let y = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(3), Dim::Const(n)]);
        let product = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::arg(1, Dtype::F32),
        );
        let fid = g
            .add(Op::L0(L0::Fold {
                carrier: add().with_lift([product]),
                axis: 1,
                acc: Dtype::F32,
                ins: smallvec::smallvec![x, y],
            }))
            .unwrap();
        assert!(fire(&mut g, fid, &STRIP).is_none());
        assert_eq!(g.chain(fid).len(), 1);
    }

    /// **The test both proposals demanded, landed before any multi-slot
    /// carrier reaches a backend:** a block-split `(m, l, o)` equals one
    /// sequential pass, and the *wrong* spelling — the outer level reusing the
    /// inner carrier, so `lift` runs on a partial accumulator — does not.
    ///
    /// This is the `accs[0]` bug one level up. At a single-slot binop the two
    /// spellings coincide, so every existing test passes straight through it.
    #[test]
    fn a_block_split_mlo_carrier_equals_one_sequential_pass() {
        let c = mlo(Dtype::F32);
        // The identity obligation first: `merge(identity, identity)` on an
        // unguarded rescale is `0 * exp((-inf) - (-inf)) = NaN`, and every
        // tree schedule merges padded identity lanes.
        assert!(c.identity_closed(&crate::carrier::PROBES));

        // Scores including a value that overflows a naive `sum(exp(s))`.
        let rows: Vec<[f32; 2]> = (0..12)
            .map(|i| {
                let s = [0.5f32, -2.0, 900.0, 3.25, -7.5, 1.0][i % 6] + i as f32 * 0.125;
                [s, (i as f32) * 0.5 - 1.0]
            })
            .collect();

        let one_pass = sequential(&c, &rows);
        // Two-pass reference, computed independently of the carrier.
        let m_ref = rows.iter().fold(f32::NEG_INFINITY, |a, r| a.max(r[0]));
        let l_ref: f32 = rows.iter().map(|r| (r[0] - m_ref).exp()).sum();
        let o_ref: f32 = rows.iter().map(|r| (r[0] - m_ref).exp() * r[1]).sum();
        assert!(one_pass.iter().all(|v| v.is_finite()), "{one_pass:?}");
        assert!((one_pass[0] - m_ref).abs() < 1e-6);
        assert!((one_pass[1] - l_ref).abs() < 1e-4 * l_ref.abs().max(1.0));
        assert!((one_pass[2] - o_ref).abs() < 1e-4 * o_ref.abs().max(1.0));

        let outer = c.as_merge();
        for blocks in [2usize, 3, 4, 6, 12] {
            let per = rows.len() / blocks;
            let partials: Vec<Vec<f32>> = rows.chunks(per).map(|b| sequential(&c, b)).collect();
            // The outer level's elements are partial ACCUMULATORS.
            let joined = partials
                .iter()
                .fold(outer.identity_f32(), |acc, p| outer.absorb(&acc, p).unwrap());
            for k in 0..3 {
                let tol = 1e-4 * one_pass[k].abs().max(1.0);
                assert!(
                    (joined[k] - one_pass[k]).abs() <= tol,
                    "{blocks} blocks, slot {k}: {} vs {}",
                    joined[k],
                    one_pass[k]
                );
            }
            // The negative control: reusing the INNER carrier at the outer
            // level applies `lift` to a partial max — `(m, 1, l)` — and the
            // answer is wrong rather than absent.
            let wrong = partials
                .iter()
                .fold(c.identity_f32(), |acc, p| c.absorb(&acc, p).unwrap());
            assert!(
                (wrong[2] - one_pass[2]).abs() > 1e-3,
                "{blocks} blocks: the wrong spelling agreed, so this test proves nothing"
            );
        }
    }

    /// ELIDE, the affine form. A reduction whose lift is the carrier's
    /// identity outside a decidable range of the reduced axis narrows to that
    /// range — ragged-batch padding, with no padding flag on any node.
    ///
    /// The work ratio is asserted against the padded extent, because
    /// "narrowed" that the emitter still walks is not narrowed.
    #[test]
    fn elide_narrows_a_padded_reduction_and_cuts_the_work() {
        let mut g = ts::graph();
        let shape = [Dim::Const(3), Dim::Const(16)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        // `select(IndexOf(1) < 6, Arg(0), 0)` — the padded row, verbatim.
        let guarded = ScalarExpr::select(
            ScalarExpr::cmp(
                CmpOp::Lt,
                ScalarExpr::index_of(1),
                ScalarExpr::lit(Splat::U32(6)),
            ),
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::lit(Splat::F32(0.0)),
        );
        let fid = ts::fold(
            &mut g,
            add().with_lift([guarded]),
            1,
            Dtype::F32,
            x,
        );
        assert!(fire(&mut g, fid, &STRIP).is_some());
        let narrowed = g
            .chain(fid)
            .into_iter()
            .find(|&m| match &g.node(m).op {
                Op::L0(L0::Fold { ins, .. }) => {
                    m != fid && g.facts(ins[0]).shape[1].known_eq(Dim::Const(6))
                }
                _ => false,
            })
            .expect("a narrowed alternative");
        let Op::L0(L0::Fold { carrier, ins, .. }) = &g.node(narrowed).op else {
            panic!()
        };
        // The predicate is gone from the lift, not merely satisfied.
        assert_eq!(carrier.lift[0].kind(), &ScalarKind::Arg(0));
        assert!(matches!(g.node(ins[0]).op, Op::L0(L0::Restride { .. })));

        // The work ratio against the padded extent: 6 of 16 elements.
        let work_of = |id: Id| {
            let node = g.node(id).clone();
            let ins: Vec<_> = node
                .children
                .iter()
                .map(|c| g.facts(*c).clone())
                .collect();
            crate::semantics::work::work_of(&node.op, &ins, g.facts(id))
        };
        let (full, cut) = (work_of(fid), work_of(narrowed));
        assert!(
            cut.macs * 16 <= full.macs * 6,
            "narrowed work {} against full {}",
            cut.macs,
            full.macs
        );

        // Numerically: the narrowed fold sums exactly the elements the mask
        // admitted, and nothing else.
        let row: Vec<f32> = (0..16).map(|i| i as f32 * 0.25 - 1.0).collect();
        let masked: f32 = row
            .iter()
            .enumerate()
            .map(|(i, v)| if i < 6 { *v } else { 0.0 })
            .sum();
        let narrowed_sum: f32 = row[..6].iter().sum();
        assert!((masked - narrowed_sum).abs() < 1e-6);
    }

    /// The same clause, a different carrier and a different identity: a `Max`
    /// reduction whose masked-out lanes are `-inf`. Nothing in the rule reads
    /// the carrier's meaning, so a mask elides against whatever identity that
    /// carrier declares.
    #[test]
    fn elide_narrows_a_max_reduction_at_its_own_identity() {
        let mut g = ts::graph();
        let shape = [Dim::Const(2), Dim::Const(32)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let max = ts::binop_carrier(BinOp::Max, Dtype::F32);
        let neg_inf = max.identity[0];
        let guarded = ScalarExpr::select(
            ScalarExpr::cmp(
                CmpOp::Ge,
                ScalarExpr::index_of(1),
                ScalarExpr::lit(Splat::U32(8)),
            ),
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::lit(neg_inf),
        );
        let fid = ts::fold(&mut g, max.with_lift([guarded]), 1, Dtype::F32, x);
        assert!(fire(&mut g, fid, &STRIP).is_some());
        let narrowed = g
            .chain(fid)
            .into_iter()
            .find(|&m| match &g.node(m).op {
                Op::L0(L0::Fold { ins, .. }) => {
                    m != fid && g.facts(ins[0]).shape[1].known_eq(Dim::Const(24))
                }
                _ => false,
            })
            .expect("a narrowed alternative");
        let Op::L0(L0::Restride { specs, .. }) = &g.node(match &g.node(narrowed).op {
            Op::L0(L0::Fold { ins, .. }) => ins[0],
            _ => unreachable!(),
        })
        .op
        else {
            panic!()
        };
        // Offset 8, size 24: the tail, not the head.
        assert!(specs[1].offset.known_eq(Dim::Const(8)));
        assert!(specs[1].size.known_eq(Dim::Const(24)));
    }

    /// The negative half. An **undecidable** predicate does not skip.
    ///
    /// Both cases here are real: a bound reading a free index is the causal
    /// mask `select(IndexOf(lk) <= IndexOf(lq) + d, .., -inf)`, which narrows
    /// the domain *per row* and no `IndexSpace` in this IR can express; a
    /// bound reading a `Uniform` is a runtime sequence length. Guessing at
    /// either is a wrong answer, so the rule declines.
    #[test]
    fn elide_declines_an_undecidable_predicate() {
        let bounds = [
            // Causal: affine in a FREE index.
            ScalarExpr::bin(
                BinOp::Add,
                ScalarExpr::index_of(0),
                ScalarExpr::lit(Splat::U32(1)),
            ),
            // A runtime scalar.
            ScalarExpr::uniform(crate::shape::SymId(7), Dtype::U32),
        ];
        for bound in bounds {
            let mut g = ts::graph();
            let shape = [Dim::Const(8), Dim::Const(16)];
            let x = ts::buffer(&mut g, Dtype::F32, &shape);
            let guarded = ScalarExpr::select(
                ScalarExpr::cmp(CmpOp::Le, ScalarExpr::index_of(1), bound.clone()),
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::lit(Splat::F32(0.0)),
            );
            let fid = ts::fold(&mut g, add().with_lift([guarded]), 1, Dtype::F32, x);
            let before = g.chain(fid).len();
            fire(&mut g, fid, &STRIP);
            // Whatever SPLIT minted, no narrowed alternative exists: every
            // member still reads the full 16-wide axis.
            for m in g.chain(fid) {
                if let Op::L0(L0::Fold { ins, .. }) = &g.node(m).op {
                    let s = g.facts(ins[0]).shape.clone();
                    assert!(
                        s.iter().all(|d| !d.known_eq(Dim::Const(6))),
                        "narrowed on an undecidable bound"
                    );
                }
            }
            assert!(g.chain(fid).len() >= before);
        }
    }

    /// A body that names the reduced coordinate is not narrowed to a window
    /// that starts anywhere but zero: narrowing renumbers that coordinate, and
    /// a positional term — ALiBi, a decay weight, a rope phase — would read
    /// the wrong index and be off by the offset rather than absent.
    #[test]
    fn elide_refuses_to_shift_a_body_that_reads_its_own_coordinate() {
        let mut g = ts::graph();
        let shape = [Dim::Const(2), Dim::Const(32)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let positional = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::cast(Dtype::F32, ScalarExpr::index_of(1)),
        );
        let guarded = ScalarExpr::select(
            ScalarExpr::cmp(
                CmpOp::Ge,
                ScalarExpr::index_of(1),
                ScalarExpr::lit(Splat::U32(8)),
            ),
            positional,
            ScalarExpr::lit(Splat::F32(0.0)),
        );
        let fid = ts::fold(&mut g, add().with_lift([guarded]), 1, Dtype::F32, x);
        fire(&mut g, fid, &STRIP);
        for m in g.chain(fid) {
            if let Op::L0(L0::Fold { ins, .. }) = &g.node(m).op {
                assert!(
                    !g.facts(ins[0]).shape[1].known_eq(Dim::Const(24)),
                    "narrowed a body that reads its own coordinate"
                );
            }
        }

        // The same body with a window that starts at zero narrows fine: the
        // coordinate is not renumbered there.
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let positional = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::cast(Dtype::F32, ScalarExpr::index_of(1)),
        );
        let guarded = ScalarExpr::select(
            ScalarExpr::cmp(
                CmpOp::Lt,
                ScalarExpr::index_of(1),
                ScalarExpr::lit(Splat::U32(8)),
            ),
            positional,
            ScalarExpr::lit(Splat::F32(0.0)),
        );
        let fid = ts::fold(&mut g, add().with_lift([guarded]), 1, Dtype::F32, x);
        fire(&mut g, fid, &STRIP);
        assert!(
            g.chain(fid).into_iter().any(|m| match &g.node(m).op {
                Op::L0(L0::Fold { ins, .. }) => g.facts(ins[0]).shape[1].known_eq(Dim::Const(8)),
                _ => false,
            }),
            "a zero-based window is still narrowed"
        );
    }

    /// A lift that is not identity-valued outside the predicate is NOT
    /// elided. A `(max, sum)` carrier's `l` slot lifts to `Lit(1)`, which is
    /// not the `Add` identity, so the guarded max slot alone cannot narrow the
    /// domain — dropping those blocks would drop a count.
    #[test]
    fn elide_refuses_a_slot_that_is_not_identity_valued() {
        let mut g = ts::graph();
        let shape = [Dim::Const(2), Dim::Const(16)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let c = crate::carrier::oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
        let guard = |body: ScalarExpr, rest: Splat| {
            ScalarExpr::select(
                ScalarExpr::cmp(
                    CmpOp::Lt,
                    ScalarExpr::index_of(1),
                    ScalarExpr::lit(Splat::U32(6)),
                ),
                body,
                ScalarExpr::lit(rest),
            )
        };
        // Slot 0 falls back to `-inf` (the Max identity); slot 1 falls back to
        // `1`, which is NOT the Add identity.
        let guarded = c.clone().with_lift([
            guard(c.lift[0].clone(), c.identity[0]),
            guard(c.lift[1].clone(), Splat::F32(1.0)),
        ]);
        let fid = ts::fold(&mut g, guarded, 1, Dtype::F32, x);
        fire(&mut g, fid, &STRIP);
        for m in g.chain(fid) {
            if let Op::L0(L0::Fold { ins, .. }) = &g.node(m).op {
                assert!(
                    !g.facts(ins[0]).shape[1].known_eq(Dim::Const(6)),
                    "narrowed a carrier whose `l` slot counts the masked lanes"
                );
            }
        }
    }

    /// Test 6. Recognition is additive at every reader count: the class holds
    /// exactly one `Fold` and one `Contract` afterwards, whether the product
    /// is read once or twice.
    #[test]
    fn recognize_contract_is_additive() {
        for extra_readers in [0usize, 1] {
            let mut g = ts::graph();
            let a = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4), Dim::Const(6)]);
            let bb = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4), Dim::Const(6)]);
            let prod = ts::map(
                &mut g,
                ScalarExpr::bin(
                    BinOp::Mul,
                    ScalarExpr::arg(0, Dtype::F32),
                    ScalarExpr::arg(1, Dtype::F32),
                ),
                &[a, bb],
            );
            let fid = ts::fold(&mut g, add(), 1, Dtype::F32, prod);
            for _ in 0..extra_readers {
                // A second reader of the product must change nothing.
                ts::map(
                    &mut g,
                    ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32)),
                    &[prod],
                );
            }
            assert!(fire(&mut g, fid, &RECOGNIZE_CONTRACT).is_some());
            let members = g.chain(fid);
            assert_eq!(members.len(), 2);
            assert!(
                members
                    .iter()
                    .any(|&m| matches!(g.node(m).op, Op::L0(L0::Fold { .. })))
            );
            assert!(
                members
                    .iter()
                    .any(|&m| matches!(g.node(m).op, Op::L0(L0::Contract { .. })))
            );
        }
    }

    #[test]
    fn contract_reassoc_regroups_a_chain() {
        let mut g = ts::graph();
        // (i,j) x (j,k) -> (i,k), then (i,k) x (k,l) -> (i,l)
        let l = |v: u8| Label(v);
        let a = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4), Dim::Const(5)]);
        let b2 = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(5), Dim::Const(6)]);
        let c = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(6), Dim::Const(7)]);
        let inner = ts::contract(
            &mut g,
            EinSpec {
                a: smallvec::smallvec![l(0), l(1)],
                b: smallvec::smallvec![l(1), l(2)],
                out: smallvec::smallvec![l(0), l(2)],
            },
            Dtype::F32,
            a,
            b2,
        );
        let outer = ts::contract(
            &mut g,
            EinSpec {
                a: smallvec::smallvec![l(0), l(2)],
                b: smallvec::smallvec![l(2), l(3)],
                out: smallvec::smallvec![l(0), l(3)],
            },
            Dtype::F32,
            inner,
            c,
        );
        assert!(fire(&mut g, outer, &CONTRACT_REASSOC).is_some());
        let members = g.chain(outer);
        assert_eq!(members.len(), 2);
        let alt = members.iter().copied().find(|&m| m != outer).unwrap();
        let Op::L0(L0::Contract { a: aa, .. }) = &g.node(alt).op else {
            panic!()
        };
        // The regrouped form contracts `a` against a fresh (b x c).
        assert_eq!(*aa, a);
    }

    #[test]
    fn const_fold_map_evaluates_a_closed_body() {
        let mut g = ts::graph();
        let anchor = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(3)]);
        let body = ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::lit(Splat::F32(2.0)),
            ScalarExpr::un(UnOp::Sqrt, ScalarExpr::lit(Splat::F32(9.0))),
        );
        let m = ts::map(&mut g, body, &[anchor]);
        assert!(fire(&mut g, m, &CONST_FOLD_MAP).is_some());
        let alt = g.chain(m).into_iter().find(|&x| x != m).unwrap();
        let Op::L0(L0::Leaf(LeafKind::Const { value, shape })) = &g.node(alt).op else {
            panic!("expected a const leaf")
        };
        assert_eq!(*value, Splat::F32(5.0));
        assert_eq!(shape.len(), 1);
    }

    #[test]
    fn identity_elim_drops_zero_adds_and_pass_through_views() {
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(3), Dim::Const(4)]);
        let view = ts::restride(
            &mut g,
            &[
                StrideSpec::dim(0, Dim::Const(3)),
                StrideSpec::dim(1, Dim::Const(4)),
            ],
            x,
        );
        let body = ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::lit(Splat::F32(0.0)),
        );
        let m = ts::map(&mut g, body, &[view]);
        assert!(fire(&mut g, m, &IDENTITY_ELIM).is_some());
        let alt = g.chain(m).into_iter().find(|&i| i != m).unwrap();
        let Op::L0(L0::Map { expr, ins, .. }) = &g.node(alt).op else {
            panic!()
        };
        assert!(matches!(expr.kind(), ScalarKind::Arg(0)));
        assert_eq!(ins[0], x);
    }

    #[test]
    fn widen_store_cast_rebuilds_at_f32_and_narrows_on_store() {
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F16, &[Dim::Const(8)]);
        let body = ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F16));
        let m = ts::map(&mut g, body, &[x]);
        assert!(fire(&mut g, m, &WIDEN_STORE_CAST).is_some());
        let alt = g.chain(m).into_iter().find(|&i| i != m).unwrap();
        let Op::L0(L0::Map { expr, .. }) = &g.node(alt).op else {
            panic!()
        };
        let ScalarKind::Cast { to, x: inner } = expr.kind() else {
            panic!("expected a narrowing store cast")
        };
        assert_eq!(*to, Dtype::F16);
        assert_eq!(inner.dtype(), Dtype::F32);
        // Firing again on the widened node is refused, so this does not grow.
        let node = g.node(alt).clone();
        let caps = ts::caps();
        let facts = g.facts_view(alt, &caps);
        let mut b = g.builder(&caps);
        assert!(widen_store_cast(&mut b, alt, &node, &facts).is_none());
    }

    #[test]
    fn unit_fold_collapse_drops_a_length_one_axis() {
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(5), Dim::Const(1)]);
        let fid = ts::fold(&mut g, add(), 1, Dtype::F32, x);
        assert!(fire(&mut g, fid, &UNIT_FOLD_COLLAPSE).is_some());
        let alt = g.chain(fid).into_iter().find(|&i| i != fid).unwrap();
        assert!(matches!(g.node(alt).op, Op::L0(L0::Restride { .. })));
        assert_eq!(g.facts(alt).shape.len(), 1);
    }

    /// Test 10. A rule fires at most once per node, so a law whose output is
    /// itself matchable terminates.
    #[test]
    fn a_rule_fires_at_most_once_per_node() {
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(2), Dim::Const(65536)]);
        let fid = ts::fold(&mut g, add(), 1, Dtype::F32, x);
        let caps = ts::caps();
        let report = CoreSaturate
            .saturate(&mut g, &caps, CORE_RULES, SaturationBudget::default())
            .unwrap();
        let fired: u32 = report
            .fired
            .iter()
            .find(|(n, _)| *n == "STRIP")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        let foldable = (0..g.len())
            .filter(|&i| {
                matches!(
                    g.node(Id(i as u32)).op,
                    Op::L0(L0::Fold { .. })
                )
            })
            .count();
        assert!(fired >= 1);
        assert!(
            fired as usize <= foldable,
            "fired {fired} times over {foldable} folds"
        );
        assert!(report.final_nodes < 200, "{}", report.final_nodes);
        let _ = fid;
    }
}
