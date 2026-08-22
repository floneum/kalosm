//! Logical algebra: the fold-splitting law, additive contraction recognition,
//! contraction reassociation, closed-expression folding, identity
//! elimination, the mixed-precision store cast and the unit-fold collapse.
//!
//! Every rule here is `Additive`: the unrewritten form stays live in the same
//! chain, and which one runs is decided once, later, by extraction.

use crate::dtype::{Dtype, RoundMode, Splat};
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::logical::{EinSpec, Label, LeafKind, Logical};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::scalar::{BinOp, CmpOp, Lit, ScalarExpr, ScalarKind, UnOp};
use crate::shape::{BoundsProof, Dim, StrideSpec};
use smallvec::SmallVec;

rule!(
    STRIP,
    level = Level::Logical,
    head = OpTag::Fold,
    tag = RuleTag::Additive,
    apply = strip,
);

rule!(
    RECOGNIZE_CONTRACT,
    level = Level::Logical,
    head = OpTag::Fold,
    tag = RuleTag::Additive,
    apply = recognize_contract,
);

rule!(
    CONTRACT_REASSOC,
    level = Level::Logical,
    head = OpTag::Contract,
    tag = RuleTag::Additive,
    apply = contract_reassoc,
);

rule!(
    CONST_FOLD_MAP,
    level = Level::Logical,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = const_fold_map,
);

rule!(
    IDENTITY_ELIM,
    level = Level::Logical,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = identity_elim,
);

rule!(
    WIDEN_STORE_CAST,
    level = Level::Logical,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = widen_store_cast,
);

rule!(
    UNIT_FOLD_COLLAPSE,
    level = Level::Logical,
    head = OpTag::Fold,
    tag = RuleTag::Additive,
    l0 = Fold { carrier, axis, ins },
    |b, id, node, f| {
        let _ = node;
        // Only a single scalar slot whose lift is the bare element: a lift
        // that computes anything still has to run, and a multi-slot carrier's
        // output carries an axis the collapse would delete.
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
        let specs: SmallVec<[StrideSpec; 6]> = (0..shape.len())
            .filter(|&j| j != axis)
            .map(|j| StrideSpec::dim(j as u32, shape[j]))
            .collect();
        let dropped = b
            .add_logical(Logical::Restride {
                specs,
                bounds: BoundsProof::Static,
                x,
            })
            .ok()?;
        b.union(id, dropped).ok()
    },
);

/// STRIP. Both clauses are minted at the same node because the driver's fired
/// set is per `(RuleId, Id)`.
///
/// * **SPLIT** — a catamorphism over a concatenation is the merge of the
///   catamorphisms over the segments.
/// * **ELIDE** — a block whose lift is identically the carrier's identity
///   contributes nothing, because `merge(acc, identity) = acc`.
pub fn strip(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    // ELIDE first: narrowing the domain makes the split cheaper.
    let elided = fold_elide(b, node, f).and_then(|x| b.union(id, x).ok());
    let split = fold_split(b, id, node, f);
    split.or(elided)
}

/// SPLIT: `Fold{c, axis}` == `Fold{c.as_merge(), axis} . Fold{c, axis+1} . block(x)`.
///
/// The carrier rides through untouched: at a contraction's summed axis this
/// is split-K, at a `(max, sum)` carrier it is online softmax, at
/// `(n, mean, m2)` it is the stable variance accumulator.
///
/// The outer level uses [`crate::carrier::Carrier::as_merge`]: its elements are partial
/// accumulators, not raw elements, and it reads ONE operand carrying the
/// inner fold's trailing carrier axis. Reusing the inner carrier applies
/// `lift` to a partial max and silently computes a wrong value; at a
/// single-slot binop the two spellings coincide.
///
/// Without the `reassoc` guard the split and unsplit forms are declared
/// value-equal and extraction may swap them on an f16 accumulator.
///
/// Every operand is blocked: one blocking view is applied to each input, and
/// inputs that do not agree on the shape it is stated against decline.
fn fold_split(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Fold {
        carrier,
        axis,
        acc,
        ins,
    }) = &node.op
    else {
        return None;
    };
    // The outer level reads partial accumulators, so the carrier must be
    // associative; `f.own().numeric` is the meet over every operand.
    if !carrier.associative || !f.own().numeric.reassoc {
        return None;
    }
    if acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }
    // A rounding lift is the QAT fake-quant value. `infer_logical` does not yet
    // derive `STRICT` from `ScalarKind::Round`, so the meet above cannot see a
    // rounding ABSORB has moved into the lift; read the carrier directly.
    if carrier.lift.iter().any(has_round) || carrier.merge.iter().any(has_round) {
        return None;
    }
    if ins.is_empty() {
        return None;
    }
    // A level that is itself a level of a split does not split again; every
    // minted fold is a fresh id the driver offers the rule again, so without
    // this the rewrite cascades.
    if ins.iter().any(|&x| stands_on_a_split(b, x, 4)) {
        return None;
    }
    let axis = *axis as usize;
    let shape = f.operand(0)?.shape.clone();
    // One blocking view serves every operand, so all operands must agree on
    // the shape it is stated against; otherwise decline.
    for i in 1..ins.len() {
        let other = &f.operand(i)?.shape;
        if other.len() != shape.len()
            || !other.iter().zip(shape.iter()).all(|(a, c)| a.known_eq(*c))
        {
            return None;
        }
    }
    // `Dim::Sym` declines: `StrideSpec::multiplier` is a `u32`, so the inner
    // extent has to be spellable.
    let extent = shape.get(axis)?.as_const()?;
    // A reduction one workgroup's lanes already cover has nowhere to put a
    // second level, so blocking it buys no parallelism and costs a dispatch.
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
            let Ok(v) = b.add_logical(Logical::Restride {
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
        let Ok(partial) = b.add_logical(Logical::Fold {
            carrier: carrier.clone(),
            axis: axis as u32 + 1,
            acc: *acc,
            ins: blocked,
        }) else {
            continue;
        };
        // The outer level's elements are partial accumulators, so it must use
        // `as_merge`, reading the one operand that carries the inner fold's
        // trailing carrier axis.
        let Ok(joined) = b.add_logical(Logical::Fold {
            carrier: carrier.as_merge(),
            axis: axis as u32,
            acc: *acc,
            ins: smallvec::smallvec![partial],
        }) else {
            continue;
        };
        // Every candidate joins the class; an un-unioned node is unreachable
        // from the root.
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
        Op::Logical(Logical::Restride {
            specs, x: inner, ..
        }) => {
            specs
                .windows(2)
                .any(|w| w[0].input_dim == w[1].input_dim && w[0].multiplier > 1)
                || stands_on_a_split(b, *inner, budget - 1)
        }
        Op::Logical(Logical::Fold { ins, .. }) => {
            ins.iter().any(|&i| stands_on_a_split(b, i, budget - 1))
        }
        _ => false,
    }
}

/// Candidate block counts for one extent: the power-of-two divisors, widest
/// first, capped at [`MAX_SPLIT_CANDIDATES`].
///
/// This stays with the Logical split rewrite: it creates new tensor algebra,
/// while `FoldDomain` schedules a Launch fold that already exists.
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
/// The narrowed range is computed from a predicate affine in `IndexOf(axis)`
/// against a closed bound. A bound that reads a free index — the causal
/// `IndexOf(lk) <= IndexOf(lq) + d` — narrows the domain per row, which no
/// `IndexSpace` in this IR can express; that case declines, and so does
/// anything `eval_closed` cannot decide.
fn fold_elide(b: &mut Builder<'_>, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Fold {
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
    // not a narrowing — so decline.
    if lo >= hi || (lo == 0 && hi == extent) {
        return None;
    }
    // Narrowing to `[lo, hi)` renumbers the reduced coordinate down by `lo`;
    // a body that names that coordinate would read the wrong index, so it
    // declines unless the window starts at zero.
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
            b.add_logical(Logical::Restride {
                specs: specs.clone(),
                bounds: BoundsProof::Static,
                x,
            })
            .ok()?,
        );
    }
    b.add_logical(Logical::Fold {
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
    // bound is the same for every row.
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

/// `Fold{Add, rank-1}(Map{mul(Arg0, Arg1)}(a, b))` also *is* a `Contract`.
///
/// The `mul`+`fold` form stays live in the same class, so a product read
/// twice keeps both options open and the extractor prices them.
pub fn recognize_contract(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Fold {
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
    // Both spellings of "multiply, then sum": the product may sit in a
    // separate `Map` the fold reads, or directly in the carrier's own `lift`
    // (what `lower_generic` mints and `ABSORB` leaves behind).
    let ins: SmallVec<[Id; 2]> = if carrier.lift[0].kind() == &ScalarKind::Arg(0) {
        let &[x] = &fold_ins[..] else {
            return None;
        };
        let Op::Logical(Logical::Map { expr, ins, outs }) = b.node(x).op.clone() else {
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
    // Read each operand through its broadcast, not around it: an operand that
    // reaches the product through a `multiplier == 0` `Restride` must be read
    // at its base's own labels, or the spec names a contraction over the
    // materialized broadcast. Naming only the axes it varies along makes the
    // spec the real einsum — `bhqd,bhkd->bhqk`.
    let (a_src, a_labels) = contract_operand(b, ins[0], rank)?;
    let (b_src, b_labels) = contract_operand(b, ins[1], rank)?;
    let contracted = Label(rank as u8 - 1);
    // The reduced axis has to be a real shared axis of both operands; where
    // one side is broadcast along it the fold is a scaled sum, not a
    // contraction.
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
        .add_logical(Logical::Contract {
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
/// is left alone: this returns the operand as given with all `rank` labels.
fn contract_operand(b: &Builder<'_>, v: Id, rank: usize) -> Option<(Id, SmallVec<[Label; 6]>)> {
    let all = || -> SmallVec<[Label; 6]> { (0..rank as u8).map(Label).collect() };
    let spine = b.trace_pure_views(v);
    if spine.views.len() != 1 {
        return Some((v, all()));
    }
    let Op::Logical(Logical::Restride { specs, .. }) = b.node(spine.views[0]).op.clone() else {
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
            // A broadcast axis: not one of this operand's labels, and
            // `Contract` will re-broadcast it from the spec.
            continue;
        }
        // Every axis this operand varies along has to be the next axis of
        // the base, read whole and in order.
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
        // Nothing was broadcast.
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
    matches!(
        (a.kind(), b.kind()),
        (ScalarKind::Arg(0), ScalarKind::Arg(1))
    ) || matches!(
        (a.kind(), b.kind()),
        (ScalarKind::Arg(1), ScalarKind::Arg(0))
    )
}

/// `Contract(Contract(a, b), c)` also equals `Contract(a, Contract(b, c))`.
///
/// Legal when the two specs share one consistent labelling and neither
/// regrouping captures a label: an inner-summed label may not reappear in
/// `c`, and an outer-summed label may not appear in `a`. Every operand must
/// permit reassociation.
pub fn contract_reassoc(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Contract {
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
    let Op::Logical(Logical::Contract {
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
        .add_logical(Logical::Contract {
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
        .add_logical(Logical::Contract {
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

/// A `Map` whose body is closed over literals alone also equals a constant
/// leaf. Additive: the unfolded form survives, so a target that would rather
/// recompute than read a splat still can.
pub fn const_fold_map(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Map { expr, outs: 1, .. }) = &node.op else {
        return None;
    };
    let value = eval_closed(expr)?;
    let folded = b
        .add_logical(Logical::Leaf(LeafKind::Const {
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
    let Op::Logical(Logical::Map { expr, ins, outs }) = &node.op else {
        return None;
    };
    let (body, body_changed) = simplify(expr);

    let mut new_ins = ins.clone();
    let mut ins_changed = false;
    for slot in new_ins.iter_mut() {
        let Op::Logical(Logical::Restride { specs, x, .. }) = b.node(*slot).op.clone() else {
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
        .add_logical(Logical::Map {
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
            Some(if to_f64(*v) != 0.0 {
                t.clone()
            } else {
                f.clone()
            })
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
    let Op::Logical(Logical::Map { expr, ins, outs }) = &node.op else {
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
        .add_logical(Logical::Map {
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
