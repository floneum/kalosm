//! Rewrite rules that recover hand-fused backwards from the composed
//! backward this crate generates.
//!
//! All three are `Additive`: the composed chain stays live in the same
//! e-class and extraction decides. Each takes `NumericContract::reassoc` as
//! its legality guard because every one re-associates float arithmetic.

use fusor2_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor2_ir::ir::logical::{EinSpec, Logical, Label};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;
use fusor2_ir::scalar::{BinOp, ScalarExpr, ScalarKind, UnOp};
use smallvec::SmallVec;

rule!(
    SOFTPLUS_BCE_ADJOINT,
    level = Level::Logical,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = softplus_bce_adjoint,
);

rule!(
    SOFTMAX_JACOBIAN,
    level = Level::Logical,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = softmax_jacobian,
);

rule!(
    ATTENTION_BACKWARD,
    level = Level::Logical,
    head = OpTag::Fold,
    tag = RuleTag::Additive,
    apply = attention_backward,
);

/// Every adjoint-recovery rule, in a fixed order (reproducibility only).
pub static ADJOINT_RULES: &[Rule] = &[SOFTPLUS_BCE_ADJOINT, SOFTMAX_JACOBIAN, ATTENTION_BACKWARD];

/// Rewrite the taped softplus-BCE chain's adjoint to the single-sigmoid form.
///
/// The composed adjoint leaves the logistic factor as `exp(u) / (1 + exp(u))`,
/// which is `inf/inf = NaN` once `exp(u)` overflows; the mathematically
/// identical `1 / (1 + exp(-u))` is not. The differentiator emits the scaled
/// spelling `(g / (1 + e)) * e`, so both forms are matched.
pub(crate) fn softplus_bce_adjoint(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.own().numeric.reassoc {
        return None;
    }
    let Op::Logical(Logical::Map { expr, ins, outs }) = &node.op else {
        return None;
    };
    let rewritten = rewrite(expr, &logistic_normalize)?;
    let alt = b
        .add_logical(Logical::Map {
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
/// with either operand order at every commutative position.
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

/// Recover the analytic softmax Jacobian from the composed backward:
/// `dS = P*dP - P*rowsum(dP*P)` -> `dS = P * (dP - rowsum(dP*P))`.
pub(crate) fn softmax_jacobian(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.own().numeric.reassoc {
        return None;
    }
    let Op::Logical(Logical::Map { expr, ins, outs }) = &node.op else {
        return None;
    };
    let rewritten = rewrite(expr, &factor_shared)?;
    let alt = b
        .add_logical(Logical::Map {
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
/// other matmul backward) is written as — `Fold{Add}(Map(Mul(a, b)))` over
/// broadcast views of two lower-rank tensors — and mint a
/// `Logical::Contract` beside it. The `mul`+`fold` form stays in the class.
pub(crate) fn attention_backward(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.own().numeric.reassoc {
        return None;
    }
    let Op::Logical(Logical::Fold {
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
    let Op::Logical(Logical::Map {
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
    // Every surviving label must be read by at least one operand.
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
        .add_logical(Logical::Contract {
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
/// `None` when the view is anything other than a pure right-order broadcast.
fn operand_labels(
    b: &Builder<'_>,
    v: Id,
    rank: usize,
) -> Option<(Id, SmallVec<[Label; 6]>)> {
    match &b.node(v).op {
        Op::Logical(Logical::Restride { specs, x, .. }) => {
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
/// `None` when nothing changed.
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
