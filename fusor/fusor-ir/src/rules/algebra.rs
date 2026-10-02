//! Logical algebra: fold splitting, contraction recognition and reassociation,
//! constant folding, identity elimination, the store cast and the unit-fold collapse.
//! Every rule is `Additive`; extraction decides.

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
        // A single scalar slot with a bare-element lift: anything else still computes
        // or carries an axis the collapse would delete.
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

/// STRIP: SPLIT (a catamorphism over a concatenation merges the segments') and ELIDE
/// (identity-lifted blocks contribute nothing), one rule since fired sets are per node.
pub fn strip(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    // ELIDE first: narrowing the domain makes the split cheaper.
    let elided = fold_elide(b, node, f).and_then(|x| b.union(id, x).ok());
    let split = fold_split(b, id, node, f);
    split.or(elided)
}

/// Split a scalar reduction into partial reductions and a final merge (needs reassoc).
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
    // Partial carriers occupy a trailing axis bound by one operand, and blocking
    // renumbers the coordinates an indexed lift reads.
    if carrier.slots.as_slice() != [crate::carrier::SlotTy::Scalar]
        || carrier.lift.iter().any(ScalarExpr::reads_index_of)
    {
        return None;
    }
    // The outer level reads partial accumulators: associativity required.
    if !carrier.associative || !f.own().numeric.reassoc {
        return None;
    }
    if acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }
    // A rounding lift (QAT fake-quant) forbids reassociation, which the numeric meet
    // cannot yet see; read the carrier directly.
    if carrier.lift.iter().any(has_round) || carrier.merge.iter().any(has_round) {
        return None;
    }
    if ins.is_empty() {
        return None;
    }
    // A level of a split does not split again, or the rewrite cascades.
    if ins.iter().any(|&x| stands_on_a_split(b, x, 4)) {
        return None;
    }
    let axis = *axis as usize;
    let shape = f.operand(0)?.shape.clone();
    // One blocking view serves every operand, so their shapes must agree.
    for i in 1..ins.len() {
        if f.operand(i)?.shape != shape {
            return None;
        }
    }
    // A symbolic extent declines: `StrideSpec::multiplier` is a `u32`.
    let extent = shape.get(axis)?.as_const()?;
    // A reduction one workgroup already covers gains nothing from a second level.
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
                bounds: if shape.iter().all(|dim| dim.as_const().is_some()) {
                    BoundsProof::Static
                } else {
                    BoundsProof::RuntimeMask
                },
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
        // The outer level reads partial accumulators via `as_merge`.
        let Ok(joined) = b.add_logical(Logical::Fold {
            carrier: carrier.as_merge(),
            axis: axis as u32,
            acc: *acc,
            ins: smallvec::smallvec![partial],
        }) else {
            continue;
        };
        // Every candidate joins the class.
        minted = b.union(id, joined).ok().or(minted);
    }
    minted
}

/// Whether a SPLIT already stands under `x`: two adjacent specs naming one `input_dim`,
/// which only [`fold_split`] mints.
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

/// Most power-of-two block counts tried per extent, widest first.
const MAX_SPLIT_CANDIDATES: usize = 3;

fn block_candidates(extent: u64) -> SmallVec<[u64; 4]> {
    [64u64, 32, 16, 8, 4, 2]
        .into_iter()
        .filter(|bl| extent.is_multiple_of(*bl) && extent / bl > 1)
        .take(MAX_SPLIT_CANDIDATES)
        .collect()
}

/// ELIDE: a reduction whose lift is the identity outside a contiguous range of the
/// reduced axis equals the reduction over that range. Per-row bounds (causal) decline.
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

    // Every slot must share one predicate and fall back to its own identity.
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
    // An empty range makes the fold a constant, not a narrowing.
    if lo >= hi || (lo == 0 && hi == extent) {
        return None;
    }
    // Narrowing renumbers the reduced coordinate by `lo`; a body reading it declines.
    if lo > 0 && bodies.iter().any(|e| e.reads_axis(*axis)) {
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

/// The contiguous range of `axis` on which `cond` is true; `None` when undecidable.
fn true_range(cond: &ScalarExpr, axis: u32, extent: u64) -> Option<(u64, u64)> {
    let ScalarKind::Cmp { op, a, b } = cond.kind() else {
        return None;
    };
    // One side names the reduced coordinate; the other must be closed.
    let (op, bound) = match (a.kind(), b.kind()) {
        (ScalarKind::IndexOf(i), _) if *i == axis => (*op, eval_closed(b)?),
        (_, ScalarKind::IndexOf(i)) if *i == axis => (flip(*op), eval_closed(a)?),
        _ => return None,
    };
    let v = bound.to_f64();
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

/// Whether `e` rounds anywhere: the marker of a value that forbids reassociation.
fn has_round(e: &ScalarExpr) -> bool {
    let mut found = false;
    e.walk(&mut |e| found |= matches!(e.kind(), ScalarKind::Round { .. }));
    found
}

/// `Fold{Add, rank-1}(Map{mul(Arg0, Arg1)}(a, b))` is also a `Contract`; both stay live.
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
    // The product sits in a `Map` the fold reads, or in the carrier's own lift.
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
    // Read each operand through a broadcasting `Restride` at its base's labels, so
    // the spec is the real einsum (`bhqd,bhkd->bhqk`).
    let (a_src, a_labels) = contract_operand(b, ins[0], rank)?;
    let (b_src, b_labels) = contract_operand(b, ins[1], rank)?;
    let contracted = Label(rank as u8 - 1);
    // The reduced axis must be shared by both operands, or it is a scaled sum.
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

/// The value a contraction reads for this operand and the labels it varies along: a
/// single broadcast-only `Restride` is its base at the base's labels; anything else is
/// left as given with all `rank` labels.
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
            // A broadcast axis: `Contract` re-broadcasts it from the spec.
            continue;
        }
        // Each varying axis must be the next base axis, read whole and in order.
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
    // Every base axis must be accounted for.
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

/// `Contract(Contract(a, b), c) == Contract(a, Contract(b, c))` when the specs share
/// one labelling, no regrouping captures a label, and every operand permits reassoc.
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
    // The inner result enters the outer under the labels it left with.
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

    // The regrouped intermediate keeps the labels something downstream needs.
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

/// A `Map` closed over literals also equals a constant leaf.
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

/// The closed interpreter `const_fold_map` runs; declines on any open leaf.
fn eval_closed(e: &ScalarExpr) -> Option<Splat> {
    let out = e.dtype();
    match e.kind() {
        ScalarKind::Lit(Lit(v)) => Some(*v),
        ScalarKind::Un { op, x } => {
            let v = eval_closed(x)?.to_f64();
            from_f64(apply_un(*op, v)?, out)
        }
        ScalarKind::Bin { op, a, b } => {
            let (x, y) = (eval_closed(a)?.to_f64(), eval_closed(b)?.to_f64());
            from_f64(apply_bin(*op, x, y, out)?, out)
        }
        ScalarKind::Cmp { op, a, b } => {
            let (x, y) = (eval_closed(a)?.to_f64(), eval_closed(b)?.to_f64());
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
            if eval_closed(c)?.to_f64() != 0.0 {
                eval_closed(t)
            } else {
                eval_closed(f)
            }
        }
        ScalarKind::Cast { to, x } => from_f64(eval_closed(x)?.to_f64(), *to),
        ScalarKind::Bitcast { to, x } => from_bits(eval_closed(x)?.bits(), *to),
        ScalarKind::Round { mode, x } => {
            let v = eval_closed(x)?.to_f64();
            from_f64(apply_round(*mode, v), out)
        }
        ScalarKind::Arg(_)
        | ScalarKind::Uniform(_)
        | ScalarKind::IndexOf(_)
        | ScalarKind::Dot { .. }
        | ScalarKind::Splat { .. } => None,
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

/// Scalar identities (`x+0`, `x*1`, `select(lit, ..)`, a no-op cast, ...) and an
/// identity-view `Restride` input, eliminated at the reading map.
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
    let mut changed = false;
    let node = match e.kind() {
        ScalarKind::Dot { .. } | ScalarKind::Splat { .. } => e.clone(),
        _ => e.map_children(&mut |child| {
            let (simpler, did_change) = simplify(child);
            changed |= did_change;
            simpler
        }),
    };
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
            Some(if v.to_f64() != 0.0 {
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
    matches!(e.kind(), ScalarKind::Lit(Lit(s)) if s.to_f64() == v)
}

/// A `Map` storing F16/BF16 also equals its arithmetic at [`Dtype::compute_dtype`] with
/// a trailing narrowing cast.
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

/// Rebuild `e` at [`Dtype::compute_dtype`]; declines on width-dependent `Bitcast`.
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
                ScalarExpr::lit(from_f64(v.to_f64(), d.compute_dtype())?)
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
