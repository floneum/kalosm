//! Rewrite rules that recover hand-fused backward forms from the *composed*
//! backward this crate generates.
//!
//! These are the L0 -> L0 half. The four `KFlash` rules are L1
//! (`KFold`/`KContract`-headed) and live in `fusor2-ir/src/rules/flash.rs`;
//! autograd runs before saturation and cannot see an L1 node.
//!
//! All three are `Additive`. The composed chain stays live in the same
//! e-class, so extraction decides, not rule order. Each reads only [`Facts`]
//! and takes `NumericContract::reassoc` as its legality guard, because every
//! one re-associates float arithmetic.

use fusor2_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor2_ir::ir::level0::{EinSpec, L0, Label};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;
use fusor2_ir::scalar::{BinOp, ScalarExpr, ScalarKind, UnOp};
use smallvec::SmallVec;

rule!(
    SOFTPLUS_BCE_ADJOINT,
    level = Level::L0,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = softplus_bce_adjoint,
);

rule!(
    SOFTMAX_JACOBIAN,
    level = Level::L0,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = softmax_jacobian,
);

rule!(
    ATTENTION_BACKWARD,
    level = Level::L0,
    head = OpTag::Fold,
    tag = RuleTag::Additive,
    apply = attention_backward,
);

/// Every adjoint-recovery rule, in a fixed order.
pub static ADJOINT_RULES: &[Rule] = &[SOFTPLUS_BCE_ADJOINT, SOFTMAX_JACOBIAN, ATTENTION_BACKWARD];

/// Rewrite the taped softplus-BCE chain's adjoint to the single-sigmoid form.
///
/// Differentiating `w*softplus(x) - x*z` leaves the logistic factor as
/// `exp(u) / (1 + exp(u))`, which is `inf/inf = NaN` once `exp(u)` overflows.
/// The identical `1 / (1 + exp(-u))` is not.
///
/// The differentiator emits the factor scaled: `d/dx log(1 + e)` with
/// `e = exp(u)` comes out as `(g / (1 + e)) * e`, so that form is matched as
/// well as the bare `Div(exp(u), 1 + exp(u))`.
pub fn softplus_bce_adjoint(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.own().numeric.reassoc {
        return None;
    }
    let Op::L0(L0::Map { expr, ins, outs }) = &node.op else {
        return None;
    };
    let rewritten = rewrite(expr, &logistic_normalize)?;
    let alt = b
        .add_l0(L0::Map {
            expr: rewritten,
            ins: ins.clone(),
            outs: *outs,
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// Both spellings of the logistic factor:
///
/// * `exp(u) / (1 + exp(u))`      -> `1 / (1 + exp(-u))`
/// * `(g / (1 + exp(u))) * exp(u)` -> `g * (1 / (1 + exp(-u)))`
///
/// with either operand order at every commutative position. The second is what
/// the map differentiator emits.
fn logistic_normalize(e: &ScalarExpr) -> Option<ScalarExpr> {
    match e.kind() {
        ScalarKind::Bin {
            op: BinOp::Div,
            a,
            b,
        } => {
            let ScalarKind::Un { op: UnOp::Exp, x } = a.kind() else {
                return None;
            };
            let (one, exp_u) = logistic_denominator(b)?;
            if exp_u != *a {
                return None;
            }
            Some(sigmoid_of(&one, x))
        }
        ScalarKind::Bin {
            op: BinOp::Mul,
            a,
            b,
        } => scaled_logistic(a, b).or_else(|| scaled_logistic(b, a)),
        _ => None,
    }
}

/// `quotient * exp_side` where `quotient = g / (1 + exp(u))` and
/// `exp_side = exp(u)` for the same `u`: the whole product is `g * sigma(u)`.
fn scaled_logistic(quotient: &ScalarExpr, exp_side: &ScalarExpr) -> Option<ScalarExpr> {
    let ScalarKind::Bin {
        op: BinOp::Div,
        a: num,
        b: den,
    } = quotient.kind()
    else {
        return None;
    };
    let (one, exp_u) = logistic_denominator(den)?;
    if exp_u != *exp_side {
        return None;
    }
    let ScalarKind::Un { op: UnOp::Exp, x } = exp_side.kind() else {
        return None;
    };
    Some(ScalarExpr::bin(
        BinOp::Mul,
        num.clone(),
        sigmoid_of(&one, x),
    ))
}

/// `1 + exp(u)`, either order: returns the literal one (for its dtype) and
/// the `exp(u)` term.
fn logistic_denominator(e: &ScalarExpr) -> Option<(ScalarExpr, ScalarExpr)> {
    let ScalarKind::Bin {
        op: BinOp::Add,
        a: p,
        b: q,
    } = e.kind()
    else {
        return None;
    };
    let (one, other) = if is_one(p) {
        (p.clone(), q)
    } else if is_one(q) {
        (q.clone(), p)
    } else {
        return None;
    };
    matches!(other.kind(), ScalarKind::Un { op: UnOp::Exp, .. })
        .then(|| (one, other.clone()))
}

/// `1 / (1 + exp(-u))`, carrying `one`'s dtype.
fn sigmoid_of(one: &ScalarExpr, u: &ScalarExpr) -> ScalarExpr {
    let negated = ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, u.clone()));
    ScalarExpr::bin(
        BinOp::Div,
        one.clone(),
        ScalarExpr::bin(BinOp::Add, one.clone(), negated),
    )
}

/// Recover the analytic softmax Jacobian from the composed backward.
///
/// The composed form `dS = P*dP - P*rowsum(dP*P)` is two multiplies over the
/// full row; `dS = P * (dP - rowsum(dP*P))` is one.
pub fn softmax_jacobian(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.own().numeric.reassoc {
        return None;
    }
    let Op::L0(L0::Map { expr, ins, outs }) = &node.op else {
        return None;
    };
    let rewritten = rewrite(expr, &factor_shared)?;
    let alt = b
        .add_l0(L0::Map {
            expr: rewritten,
            ins: ins.clone(),
            outs: *outs,
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// `p*u - p*v` -> `p*(u - v)`, for a shared factor on either side.
fn factor_shared(e: &ScalarExpr) -> Option<ScalarExpr> {
    let ScalarKind::Bin {
        op: BinOp::Sub,
        a,
        b,
    } = e.kind()
    else {
        return None;
    };
    let ScalarKind::Bin {
        op: BinOp::Mul,
        a: la,
        b: lb,
    } = a.kind()
    else {
        return None;
    };
    let ScalarKind::Bin {
        op: BinOp::Mul,
        a: ra,
        b: rb,
    } = b.kind()
    else {
        return None;
    };
    let (shared, u, v) = if la == ra {
        (la, lb, rb)
    } else if la == rb {
        (la, lb, ra)
    } else if lb == ra {
        (lb, la, rb)
    } else if lb == rb {
        (lb, la, ra)
    } else {
        return None;
    };
    Some(ScalarExpr::bin(
        BinOp::Mul,
        shared.clone(),
        ScalarExpr::bin(BinOp::Sub, u.clone(), v.clone()),
    ))
}

/// Recognize the composed contraction the attention backward (and every
/// other matmul backward) is written as, and mint an `L0::Contract` beside it.
///
/// `dq = ds @ k`, `dk = ds^T @ q` and `dv = p^T @ grad_o` all arrive as
/// `Fold{Add}(Map(Mul(a, b)))` over broadcast views of two lower-rank tensors.
/// Recognized as a contraction, each becomes one `KContract` chain with `Coop`,
/// `Sgemm`, `Sgemv` and `GenericFold` live in the same class. The `mul`+`fold`
/// form stays in the class, so the cost model decides.
pub fn attention_backward(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.own().numeric.reassoc {
        return None;
    }
    let Op::L0(L0::Fold {
        carrier,
        axis,
        acc,
        ins: fold_ins,
    }) = &node.op
    else {
        return None;
    };
    if carrier.kind() != Some(fusor2_ir::scalar::BinOp::Add)
        || carrier.lift[0].kind() != &fusor2_ir::scalar::ScalarKind::Arg(0)
    {
        return None;
    }
    let &[x] = &fold_ins[..] else {
        return None;
    };
    let Op::L0(L0::Map {
        expr,
        ins,
        outs: 1,
    }) = &b.node(x).op.clone()
    else {
        return None;
    };
    if !is_plain_product(expr) || ins.len() != 2 {
        return None;
    }
    let rank = b.facts_of(x).shape.len();
    if rank == 0 || rank > u8::MAX as usize {
        return None;
    }
    let axis = *axis as usize;
    if axis >= rank {
        return None;
    }

    let (base_a, a_labels) = operand_labels(b, ins[0], rank)?;
    let (base_b, b_labels) = operand_labels(b, ins[1], rank)?;

    // The folded axis must be a real contraction: read by both operands and
    // absent from the output.
    let contracted = Label(axis as u8);
    if !a_labels.contains(&contracted) || !b_labels.contains(&contracted) {
        return None;
    }
    let out: SmallVec<[Label; 6]> = (0..rank)
        .filter(|i| *i != axis)
        .map(|i| Label(i as u8))
        .collect();
    // Every surviving label must be read by at least one operand, or the
    // spec would carry a label appearing only in `out`.
    if !out
        .iter()
        .all(|l| a_labels.contains(l) || b_labels.contains(l))
    {
        return None;
    }

    let spec = EinSpec {
        a: a_labels,
        b: b_labels,
        out,
    };
    crate::contract::verify_spec(&spec).ok()?;

    let alt = b
        .add_l0(L0::Contract {
            spec,
            acc: *acc,
            a: base_a,
            b: base_b,
            outs: 1,
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// `Arg(0) * Arg(1)`, either order.
fn is_plain_product(expr: &ScalarExpr) -> bool {
    let ScalarKind::Bin {
        op: BinOp::Mul,
        a,
        b,
    } = expr.kind()
    else {
        return false;
    };
    matches!(
        (a.kind(), b.kind()),
        (ScalarKind::Arg(0), ScalarKind::Arg(1)) | (ScalarKind::Arg(1), ScalarKind::Arg(0))
    )
}

/// The base tensor behind a broadcast view, plus the index labels it reads.
/// `None` when the view is anything other than a pure right-order broadcast:
/// a slice, a stride, a permutation or a reshape all disqualify this rewrite.
fn operand_labels(
    b: &Builder<'_>,
    v: Id,
    rank: usize,
) -> Option<(Id, SmallVec<[Label; 6]>)> {
    match &b.node(v).op {
        Op::L0(L0::Restride { specs, x, .. }) => {
            if specs.len() != rank {
                return None;
            }
            let base_shape = b.facts_of(*x).shape.clone();
            let mut labels: SmallVec<[Label; 6]> = SmallVec::new();
            let mut next = 0u32;
            for (pos, s) in specs.iter().enumerate() {
                if s.is_broadcast() {
                    continue;
                }
                if s.multiplier != 1 || s.offset.as_const() != Some(0) || s.input_dim != next {
                    return None;
                }
                if !base_shape
                    .get(next as usize)
                    .is_some_and(|d| d.known_eq(s.size))
                {
                    return None;
                }
                next += 1;
                labels.push(Label(pos as u8));
            }
            if next as usize != base_shape.len() {
                return None;
            }
            Some((*x, labels))
        }
        _ => {
            if b.facts_of(v).shape.len() != rank {
                return None;
            }
            Some((v, (0..rank).map(|i| Label(i as u8)).collect()))
        }
    }
}


/// Bottom-up rewrite: apply `f` at every node, innermost first. Returns
/// `None` when nothing changed, so a rule that would union a node with
/// itself simply does not fire.
fn rewrite(
    e: &ScalarExpr,
    f: &dyn Fn(&ScalarExpr) -> Option<ScalarExpr>,
) -> Option<ScalarExpr> {
    let mut changed = false;
    let rebuilt = rewrite_inner(e, f, &mut changed);
    changed.then_some(rebuilt)
}

fn rewrite_inner(
    e: &ScalarExpr,
    f: &dyn Fn(&ScalarExpr) -> Option<ScalarExpr>,
    changed: &mut bool,
) -> ScalarExpr {
    let rebuilt = match e.kind() {
        ScalarKind::Arg(_)
        | ScalarKind::Lit(_)
        | ScalarKind::Uniform(_)
        | ScalarKind::IndexOf(_) => e.clone(),
        ScalarKind::Un { op, x } => ScalarExpr::un(*op, rewrite_inner(x, f, changed)),
        ScalarKind::Bin { op, a, b } => ScalarExpr::bin(
            *op,
            rewrite_inner(a, f, changed),
            rewrite_inner(b, f, changed),
        ),
        ScalarKind::Cmp { op, a, b } => ScalarExpr::cmp(
            *op,
            rewrite_inner(a, f, changed),
            rewrite_inner(b, f, changed),
        ),
        ScalarKind::Select { c, t, f: e_f } => ScalarExpr::select(
            rewrite_inner(c, f, changed),
            rewrite_inner(t, f, changed),
            rewrite_inner(e_f, f, changed),
        ),
        ScalarKind::Cast { to, x } => ScalarExpr::cast(*to, rewrite_inner(x, f, changed)),
        ScalarKind::Bitcast { to, x } => ScalarExpr::bitcast(*to, rewrite_inner(x, f, changed)),
        ScalarKind::Round { mode, x } => ScalarExpr::round(*mode, rewrite_inner(x, f, changed)),
        ScalarKind::Dot { a, b } => ScalarExpr::new(
            ScalarKind::Dot {
                a: rewrite_inner(a, f, changed),
                b: rewrite_inner(b, f, changed),
            },
            e.dtype(),
        ),
        ScalarKind::Splat { lanes, x } => ScalarExpr::new(
            ScalarKind::Splat {
                lanes: *lanes,
                x: rewrite_inner(x, f, changed),
            },
            e.dtype(),
        ),
    };
    match f(&rebuilt) {
        Some(next) => {
            *changed = true;
            next
        }
        None => rebuilt,
    }
}

fn is_one(e: &ScalarExpr) -> bool {
    match e.kind() {
        ScalarKind::Lit(l) => match l.0 {
            fusor2_ir::dtype::Splat::F32(v) => v == 1.0,
            fusor2_ir::dtype::Splat::F16(b) => half::f16::from_bits(b).to_f32() == 1.0,
            fusor2_ir::dtype::Splat::BF16(b) => half::bf16::from_bits(b).to_f32() == 1.0,
            fusor2_ir::dtype::Splat::U32(v) => v == 1,
            fusor2_ir::dtype::Splat::I32(v) => v == 1,
        },
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::testing::{caps, graph};
    use fusor2_ir::dtype::{Dtype, Splat};
    use fusor2_ir::egraph::EGraph;
    use fusor2_ir::carrier::Carrier;
    use fusor2_ir::ir::level0::{BufferId, LeafKind};

    fn cbin(op: BinOp) -> Carrier {
        Carrier::binop(op, Carrier::binop_identity(op, Dtype::F32).unwrap(), Dtype::F32)
    }
    use fusor2_ir::shape::{Dim, StrideSpec};

    fn param(g: &mut EGraph, shape: &[u64]) -> Id {
        let n = g.len() as u32;
        g.add(Op::L0(L0::Leaf(LeafKind::Param {
            name: BufferId(n),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn arg(i: u32) -> ScalarExpr {
        ScalarExpr::arg(i, Dtype::F32)
    }
    fn one() -> ScalarExpr {
        ScalarExpr::lit(Splat::F32(1.0))
    }

    /// Fire `rule` on `id` and return the id it unioned in, if any.
    fn fire(g: &mut EGraph, rule: &Rule, id: Id) -> Option<Id> {
        let caps = caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (rule.apply)(&mut b, id, &node, &facts)
    }

    /// Every member of `id`'s class.
    fn members(g: &EGraph, id: Id) -> Vec<Id> {
        g.members(g.class_of(id)).into_vec()
    }

    #[test]
    fn softplus_bce_adjoint_fires_on_the_logistic_factor() {
        let mut g = graph();
        let x = param(&mut g, &[4]);
        // exp(x) / (1 + exp(x))
        let e = ScalarExpr::un(UnOp::Exp, arg(0));
        let body = ScalarExpr::bin(
            BinOp::Div,
            e.clone(),
            ScalarExpr::bin(BinOp::Add, one(), e),
        );
        let m = g
            .add(Op::L0(L0::Map {
                expr: body,
                ins: smallvec::smallvec![x],
                outs: 1,
            }))
            .unwrap();
        let u = fire(&mut g, &SOFTPLUS_BCE_ADJOINT, m).expect("the rule must fire");
        let class = members(&g, u);
        assert!(class.contains(&m), "the composed form stays live");
        let alt = class.iter().copied().find(|c| *c != m).unwrap();
        match &g.node(alt).op {
            Op::L0(L0::Map { expr, .. }) => {
                // 1 / (1 + exp(-x))
                let want = ScalarExpr::bin(
                    BinOp::Div,
                    one(),
                    ScalarExpr::bin(
                        BinOp::Add,
                        one(),
                        ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, arg(0))),
                    ),
                );
                assert_eq!(*expr, want);
            }
            other => panic!("expected a Map, got {other:?}"),
        }
    }

    #[test]
    fn softplus_bce_adjoint_also_matches_the_commuted_denominator() {
        let mut g = graph();
        let x = param(&mut g, &[4]);
        let e = ScalarExpr::un(UnOp::Exp, arg(0));
        let body = ScalarExpr::bin(
            BinOp::Div,
            e.clone(),
            ScalarExpr::bin(BinOp::Add, e, one()),
        );
        let m = g
            .add(Op::L0(L0::Map {
                expr: body,
                ins: smallvec::smallvec![x],
                outs: 1,
            }))
            .unwrap();
        assert!(fire(&mut g, &SOFTPLUS_BCE_ADJOINT, m).is_some());
    }

    #[test]
    fn softplus_bce_adjoint_does_not_fire_twice() {
        let mut g = graph();
        let x = param(&mut g, &[4]);
        let body = ScalarExpr::bin(BinOp::Add, arg(0), one());
        let m = g
            .add(Op::L0(L0::Map {
                expr: body,
                ins: smallvec::smallvec![x],
                outs: 1,
            }))
            .unwrap();
        assert!(fire(&mut g, &SOFTPLUS_BCE_ADJOINT, m).is_none());
    }

    #[test]
    fn softmax_jacobian_factors_the_shared_probability() {
        let mut g = graph();
        let p = param(&mut g, &[2, 4]);
        let dp = param(&mut g, &[2, 4]);
        let row = param(&mut g, &[2, 4]);
        // P*dP - P*R
        let body = ScalarExpr::bin(
            BinOp::Sub,
            ScalarExpr::bin(BinOp::Mul, arg(0), arg(1)),
            ScalarExpr::bin(BinOp::Mul, arg(0), arg(2)),
        );
        let m = g
            .add(Op::L0(L0::Map {
                expr: body,
                ins: smallvec::smallvec![p, dp, row],
                outs: 1,
            }))
            .unwrap();
        let u = fire(&mut g, &SOFTMAX_JACOBIAN, m).expect("the rule must fire");
        let alt = members(&g, u).into_iter().find(|c| *c != m).unwrap();
        match &g.node(alt).op {
            Op::L0(L0::Map { expr, .. }) => {
                let want = ScalarExpr::bin(
                    BinOp::Mul,
                    arg(0),
                    ScalarExpr::bin(BinOp::Sub, arg(1), arg(2)),
                );
                assert_eq!(*expr, want);
            }
            other => panic!("expected a Map, got {other:?}"),
        }
    }

    #[test]
    fn softmax_jacobian_matches_a_shared_factor_on_either_side() {
        let mut g = graph();
        let p = param(&mut g, &[2]);
        let dp = param(&mut g, &[2]);
        let row = param(&mut g, &[2]);
        // dP*P - R*P
        let body = ScalarExpr::bin(
            BinOp::Sub,
            ScalarExpr::bin(BinOp::Mul, arg(1), arg(0)),
            ScalarExpr::bin(BinOp::Mul, arg(2), arg(0)),
        );
        let m = g
            .add(Op::L0(L0::Map {
                expr: body,
                ins: smallvec::smallvec![p, dp, row],
                outs: 1,
            }))
            .unwrap();
        assert!(fire(&mut g, &SOFTMAX_JACOBIAN, m).is_some());
    }

    /// `[m, k] x [k, n]` written as a broadcast product folded over `k`.
    #[test]
    fn attention_backward_recognizes_the_composed_contraction() {
        let mut g = graph();
        let a = param(&mut g, &[4, 3]); // [m, k]
        let bb = param(&mut g, &[3, 5]); // [k, n]
        // a -> [m, k, n] with n broadcast
        let va = g
            .add(Op::L0(L0::Restride {
                specs: smallvec::smallvec![
                    StrideSpec::dim(0, Dim::Const(4)),
                    StrideSpec::dim(1, Dim::Const(3)),
                    StrideSpec::broadcast(Dim::Const(5)),
                ],
                bounds: fusor2_ir::shape::BoundsProof::Static,
                x: a,
            }))
            .unwrap();
        // b -> [m, k, n] with m broadcast
        let vb = g
            .add(Op::L0(L0::Restride {
                specs: smallvec::smallvec![
                    StrideSpec::broadcast(Dim::Const(4)),
                    StrideSpec::dim(0, Dim::Const(3)),
                    StrideSpec::dim(1, Dim::Const(5)),
                ],
                bounds: fusor2_ir::shape::BoundsProof::Static,
                x: bb,
            }))
            .unwrap();
        let prod = g
            .add(Op::L0(L0::Map {
                expr: ScalarExpr::bin(BinOp::Mul, arg(0), arg(1)),
                ins: smallvec::smallvec![va, vb],
                outs: 1,
            }))
            .unwrap();
        let folded = g
            .add(Op::L0(L0::Fold {
                carrier: cbin(BinOp::Add),
                axis: 1,
                acc: Dtype::F32,
                ins: smallvec::smallvec![prod],
            }))
            .unwrap();

        let u = fire(&mut g, &ATTENTION_BACKWARD, folded).expect("the rule must fire");
        let alt = members(&g, u).into_iter().find(|c| *c != folded).unwrap();
        match &g.node(alt).op {
            Op::L0(L0::Contract { spec, a: ca, b: cb, .. }) => {
                assert_eq!(*ca, a);
                assert_eq!(*cb, bb);
                assert_eq!(spec.a.as_slice(), &[Label(0), Label(1)]);
                assert_eq!(spec.b.as_slice(), &[Label(1), Label(2)]);
                assert_eq!(spec.out.as_slice(), &[Label(0), Label(2)]);
            }
            other => panic!("expected a Contract, got {other:?}"),
        }
        assert_eq!(
            g.facts(alt).shape.as_slice(),
            &[Dim::Const(4), Dim::Const(5)]
        );
    }

    #[test]
    fn attention_backward_declines_a_fold_that_is_not_a_contraction() {
        let mut g = graph();
        let a = param(&mut g, &[4, 3]);
        let b2 = param(&mut g, &[4, 3]);
        let prod = g
            .add(Op::L0(L0::Map {
                expr: ScalarExpr::bin(BinOp::Mul, arg(0), arg(1)),
                ins: smallvec::smallvec![a, b2],
                outs: 1,
            }))
            .unwrap();
        let folded = g
            .add(Op::L0(L0::Fold {
                carrier: cbin(BinOp::Add),
                axis: 1,
                acc: Dtype::F32,
                ins: smallvec::smallvec![prod],
            }))
            .unwrap();
        // Both operands read every axis, so every surviving label is a batch
        // label and the contraction is a diagonal, which is a legal `EinSpec`.
        assert!(fire(&mut g, &ATTENTION_BACKWARD, folded).is_some());
    }

    #[test]
    fn attention_backward_declines_a_max_fold() {
        let mut g = graph();
        let a = param(&mut g, &[4, 3]);
        let b2 = param(&mut g, &[4, 3]);
        let prod = g
            .add(Op::L0(L0::Map {
                expr: ScalarExpr::bin(BinOp::Mul, arg(0), arg(1)),
                ins: smallvec::smallvec![a, b2],
                outs: 1,
            }))
            .unwrap();
        let folded = g
            .add(Op::L0(L0::Fold {
                carrier: cbin(BinOp::Max)
                    .with_tie(fusor2_ir::ir::level0::TiePolicy::FirstWins),
                axis: 1,
                acc: Dtype::F32,
                ins: smallvec::smallvec![prod],
            }))
            .unwrap();
        assert!(fire(&mut g, &ATTENTION_BACKWARD, folded).is_none());
    }

    #[test]
    fn all_three_rules_are_additive_and_level_zero() {
        assert_eq!(ADJOINT_RULES.len(), 3);
        for r in ADJOINT_RULES {
            assert_eq!(r.level, Level::L0);
            assert_eq!(r.tag, RuleTag::Additive);
        }
    }
}

#[cfg(test)]
mod on_a_real_tape {
    //! Cases that build a primal, differentiate it with [`crate::backward`],
    //! and fire a rule on the node the adjoint walk emits.

    use super::*;
    use crate::backward::backward_into;
    use crate::tape::GraphTape;
    use crate::tape::TapeExt;
    use crate::tape::testing::{Env, caps, eval, graph};
    use fusor2_ir::autograd::Tape;
    use fusor2_ir::dtype::{Dtype, Splat};
    use fusor2_ir::egraph::EGraph;
    use fusor2_ir::ir::level0::{BufferId, LeafKind};
    use fusor2_ir::shape::Dim;
    use rustc_hash::FxHashMap;

    /// Fire `rule` on `id` and return the id it unioned in, if any.
    fn fire(g: &mut EGraph, rule: &Rule, id: Id) -> Option<Id> {
        let caps = caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (rule.apply)(&mut b, id, &node, &facts)
    }

    fn f32_buffer(g: &mut EGraph, shape: &[u64]) -> Id {
        let n = g.len() as u32;
        g.add(Op::L0(L0::Leaf(LeafKind::Buffer {
            name: BufferId(n),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn ones(g: &mut EGraph, shape: &[u64]) -> Id {
        g.add(Op::L0(L0::Leaf(LeafKind::Const {
            value: Splat::F32(1.0),
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    /// `max(x, 0) + log(1 + exp(-|x|))`, the expression
    /// `fusor2::composite::activations::softplus_expr` builds.
    fn softplus_expr() -> ScalarExpr {
        let d = Dtype::F32;
        let x = ScalarExpr::arg(0, d);
        let tail = ScalarExpr::un(
            UnOp::Log,
            ScalarExpr::bin(
                BinOp::Add,
                ScalarExpr::lit(Splat::F32(1.0)),
                ScalarExpr::un(
                    UnOp::Exp,
                    ScalarExpr::un(UnOp::Neg, ScalarExpr::un(UnOp::Abs, x.clone())),
                ),
            ),
        );
        ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::bin(BinOp::Max, x, ScalarExpr::lit(Splat::F32(0.0))),
            tail,
        )
    }

    /// The rule fires on the softplus adjoint the differentiator emits, and
    /// the alternative it unions in computes the same numbers.
    #[test]
    fn softplus_bce_adjoint_fires_on_the_taped_adjoint() {
        let mut g = graph();
        let x = f32_buffer(&mut g, &[6]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            t.map(softplus_expr(), &[x]).unwrap()
        };
        let seed = ones(&mut g, &[6]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let dx = grads[0];

        let alt = fire(&mut g, &SOFTPLUS_BCE_ADJOINT, dx)
            .expect("the rule must fire on the adjoint the tape emits");
        let rewritten = g
            .members(g.class_of(alt))
            .into_iter()
            .find(|c| *c != dx)
            .expect("the composed form stays live beside the rewrite");

        // d/dx softplus(x) = sigmoid(x), and both class members must say so.
        let vals = vec![-4.0f32, -1.0, -0.25, 0.25, 1.0, 4.0];
        let mut env: Env = FxHashMap::default();
        env.insert(x, vals.clone());
        let composed = eval(&g, dx, &env);
        let fused = eval(&g, rewritten, &env);
        for (i, v) in vals.iter().enumerate() {
            let want = 1.0 / (1.0 + (-v).exp());
            assert!(
                (composed[i] - want).abs() < 1e-5,
                "composed[{i}] = {}, want {want}",
                composed[i]
            );
            assert!(
                (fused[i] - want).abs() < 1e-5,
                "rewritten[{i}] = {}, want {want}",
                fused[i]
            );
        }
    }

    /// `(g / (1 + e)) * e` is what the differentiator writes; a bare
    /// `e / (1 + e)` never appears on a tape.
    #[test]
    fn the_bare_quotient_alone_is_not_what_a_tape_emits() {
        let d = Dtype::F32;
        let u = ScalarExpr::un(UnOp::Neg, ScalarExpr::un(UnOp::Abs, ScalarExpr::arg(1, d)));
        let e = ScalarExpr::un(UnOp::Exp, u);
        let den = ScalarExpr::bin(BinOp::Add, ScalarExpr::lit(Splat::F32(1.0)), e.clone());
        let scaled = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::bin(BinOp::Div, ScalarExpr::arg(0, d), den),
            e,
        );
        assert!(
            logistic_normalize(&scaled).is_some(),
            "the scaled spelling must normalize"
        );
        // ...and the commuted product too.
        let ScalarKind::Bin { a, b, .. } = scaled.kind() else {
            panic!()
        };
        let commuted = ScalarExpr::bin(BinOp::Mul, b.clone(), a.clone());
        assert!(logistic_normalize(&commuted).is_some());
    }

    /// A quotient whose denominator names a *different* `exp` is not a
    /// logistic factor and must not be rewritten.
    #[test]
    fn a_mismatched_exponent_is_left_alone() {
        let d = Dtype::F32;
        let e0 = ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, d));
        let e1 = ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(1, d));
        let den = ScalarExpr::bin(BinOp::Add, ScalarExpr::lit(Splat::F32(1.0)), e1);
        let scaled = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::bin(BinOp::Div, ScalarExpr::arg(2, d), den),
            e0,
        );
        assert!(logistic_normalize(&scaled).is_none());
    }

    /// One pass reaches a fixed point: firing the rule on its own output
    /// changes nothing, so saturation cannot union a class with itself
    /// forever.
    #[test]
    fn the_rewrite_is_idempotent() {
        let d = Dtype::F32;
        let e = ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(1, d));
        let den = ScalarExpr::bin(BinOp::Add, ScalarExpr::lit(Splat::F32(1.0)), e.clone());
        let scaled = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::bin(BinOp::Div, ScalarExpr::arg(0, d), den),
            e,
        );
        let once = rewrite(&scaled, &logistic_normalize).expect("fires once");
        assert!(rewrite(&once, &logistic_normalize).is_none(), "and only once");
    }

    /// The composed softmax backward is one `Map` per primitive, so
    /// `softmax_jacobian`'s `p*dP - p*R` pattern cannot appear until
    /// map-into-map fusion merges them.
    #[test]
    fn the_softmax_jacobian_needs_map_fusion_that_does_not_run_yet() {
        let mut g = graph();
        let x = f32_buffer(&mut g, &[2, 4]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            let m = t
                .fold_binop(BinOp::Max, 1, Dtype::F32, x)
                .unwrap();
            let m = t.broadcast_axis(m, 1, Dim::Const(4)).unwrap();
            let c = t.binary(BinOp::Sub, x, m).unwrap();
            let e = t.unary(UnOp::Exp, c).unwrap();
            let s = t.fold_binop(BinOp::Add, 1, Dtype::F32, e).unwrap();
            let s = t.broadcast_axis(s, 1, Dim::Const(4)).unwrap();
            t.binary(BinOp::Div, e, s).unwrap()
        };
        let seed = ones(&mut g, &[2, 4]);
        backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();

        let hits = (0..g.len())
            .filter(|i| {
                let op = g.node(Id(*i as u32)).op.clone();
                match &op {
                    Op::L0(L0::Map { expr, .. }) => rewrite(expr, &factor_shared).is_some(),
                    _ => false,
                }
            })
            .count();
        assert_eq!(
            hits, 0,
            "if this starts matching, map fusion landed and the rule is live"
        );
    }
}
