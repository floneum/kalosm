//! The `ScalarExpr` differentiator — the one function that covers the whole
//! elementwise / comparison / activation surface.
//!
//! One tree walk covers all 23 elementwise unaries, all 12 comparisons,
//! `where_cond`, `clamp`, the 8 scalar-arith unaries, `cast` in both
//! directions, `round`, `relu`, `sigmoid`, `silu`, `gelu` and `tanh_exact`.
//! There is no per-op `match` outside [`ScalarKind`].
//!
//! Derivatives are written in terms of the operand, never the primal
//! output: `d(exp)/dx = s * exp(x)`, not `s * out`. `ScalarExpr` is
//! hash-consed, so the recomputed `exp(x)` is one term inside the same
//! kernel body rather than an extra operand edge.

use fusor2_ir::autograd::{Grads, Tape, Val};
use fusor2_ir::dtype::{Dtype, Splat};
use fusor2_ir::ir::Node;
use fusor2_ir::ir::logical::Logical;
use fusor2_ir::ir::Op;
use fusor2_ir::scalar::{BinOp, CmpOp, ScalarExpr, ScalarKind, UnOp};
use fusor2_ir::{Error, Result};
use smallvec::SmallVec;

/// One partial per `Arg(i)` of the differentiated body.
pub type Partials = SmallVec<[Option<ScalarExpr>; 4]>;

/// Adjoint of `Logical::Map`: differentiate the body once with respect to each
/// `Arg(i)`, then map the resulting expression over `(grad, inputs...)`.
///
/// Inside [`differentiate`]'s result, `Arg(i)` still denotes primal operand
/// `i` and `Arg(nargs)` denotes the incoming gradient. This function
/// **rebases**: each emitted body is composed onto a compact operand list
/// holding only the slots that partial actually reads, with the gradient
/// first. Unused operands are dropped from `ins`.
pub fn map_adjoint(
    tape: &mut dyn Tape,
    node: &Node,
    grad: Val,
    ins: &[Val],
    _out: Val,
) -> Result<Grads> {
    let Op::Logical(Logical::Map { expr, outs, .. }) = &node.op else {
        return Err(Error::Plan(format!(
            "map_adjoint called on a non-Map node: {:?}",
            node.op
        )));
    };
    if *outs != 1 {
        // `ScalarExpr` has exactly one result, so a `Map` with `outs > 1`
        // carries no body describing its extra slots.
        return Err(Error::Plan(
            "multi-output Map has no per-slot body to differentiate".into(),
        ));
    }
    let nargs = ins.len() as u32;
    let seed_dtype = expr.dtype();
    let seed = ScalarExpr::arg(nargs, seed_dtype);
    let partials = differentiate(expr, &seed, nargs);

    let mut grads: Grads = SmallVec::with_capacity(ins.len());
    for (slot, partial) in partials.into_iter().enumerate() {
        let target = ins[slot];
        let Some(partial) = partial else {
            grads.push(None);
            continue;
        };
        // A structurally-zero partial is a zero tensor, not a kernel.
        if is_zero(&partial) {
            grads.push(Some(tape.zeros_like(target)?));
            continue;
        }
        let (body, operands) = rebase(&partial, nargs, grad, ins, tape);
        let g = tape.map(body, &operands)?;
        // The gradient of operand `slot` is shaped like operand `slot`; the
        // Map invariant already forces every operand to the output shape,
        // so no reduction is possible or needed here.
        grads.push(Some(g));
    }
    Ok(grads)
}

/// Rebuild `partial` over a compact operand list: `Arg(0)` is the incoming
/// gradient when it is read, then each primal operand the partial reads, in
/// slot order.
fn rebase(
    partial: &ScalarExpr,
    nargs: u32,
    grad: Val,
    ins: &[Val],
    tape: &dyn Tape,
) -> (ScalarExpr, SmallVec<[Val; 4]>) {
    let mut used = vec![false; nargs as usize + 1];
    mark_used(partial, &mut used);

    let mut operands: SmallVec<[Val; 4]> = SmallVec::new();
    let mut args: Vec<ScalarExpr> = Vec::with_capacity(nargs as usize + 1);
    // Placeholder entries; overwritten below for the slots that are used.
    for i in 0..=nargs {
        args.push(ScalarExpr::arg(i, Dtype::F32));
    }
    if used[nargs as usize] {
        args[nargs as usize] = ScalarExpr::arg(0, tape.facts(grad).dtype);
        operands.push(grad);
    }
    for slot in 0..nargs as usize {
        if used[slot] {
            let next = operands.len() as u32;
            args[slot] = ScalarExpr::arg(next, tape.facts(ins[slot]).dtype);
            operands.push(ins[slot]);
        }
    }
    if operands.is_empty() {
        // A partial reading nothing is still shaped like the output; give it
        // the gradient as an unread operand to fix the index space.
        operands.push(grad);
    }
    (partial.compose(&args), operands)
}

fn mark_used(e: &ScalarExpr, used: &mut [bool]) {
    match e.kind() {
        ScalarKind::Arg(i) => {
            if let Some(slot) = used.get_mut(*i as usize) {
                *slot = true;
            }
        }
        ScalarKind::Lit(_) | ScalarKind::Uniform(_) | ScalarKind::IndexOf(_) => {}
        ScalarKind::Un { x, .. }
        | ScalarKind::Cast { x, .. }
        | ScalarKind::Bitcast { x, .. }
        | ScalarKind::Round { x, .. }
        | ScalarKind::Splat { x, .. } => mark_used(x, used),
        ScalarKind::Bin { a, b, .. } | ScalarKind::Cmp { a, b, .. } | ScalarKind::Dot { a, b } => {
            mark_used(a, used);
            mark_used(b, used);
        }
        ScalarKind::Select { c, t, f } => {
            mark_used(c, used);
            mark_used(t, used);
            mark_used(f, used);
        }
    }
}

/// `d(expr)/d(Arg(i))` for every `i < nargs`, seeded with `seed`.
///
/// Walks top-down carrying the incoming scalar adjoint and accumulates one
/// partial per `Arg(i)`. Every branch is visited even when its local
/// derivative is zero, so an `Arg` reachable only through a comparison still
/// receives an (identically zero) partial.
pub fn differentiate(expr: &ScalarExpr, seed: &ScalarExpr, nargs: u32) -> Partials {
    let mut out: Partials = smallvec::smallvec![None; nargs as usize];
    walk(expr, seed.clone(), &mut out);
    out
}

fn accumulate_into(out: &mut Partials, slot: usize, term: ScalarExpr) {
    if let Some(entry) = out.get_mut(slot) {
        *entry = Some(match entry.take() {
            Some(prev) => add(prev, term),
            None => term,
        });
    }
}

fn walk(expr: &ScalarExpr, seed: ScalarExpr, out: &mut Partials) {
    match expr.kind() {
        ScalarKind::Arg(i) => accumulate_into(out, *i as usize, seed),
        ScalarKind::Lit(_) | ScalarKind::Uniform(_) | ScalarKind::IndexOf(_) => {}

        ScalarKind::Un { op, x } => {
            let d = unary_derivative(*op, x, &seed);
            match d {
                Some(term) => walk(x, term, out),
                // `Unpack2x16Float` reads a `u32`; there is no gradient into
                // an integer bit pattern. Seed the subtree with zero so any
                // `Arg` below it still receives a partial.
                None => walk(x, zero_like(x), out),
            }
        }

        ScalarKind::Bin { op, a, b } => {
            let (da, db) = binary_derivative(*op, a, b, &seed);
            walk(a, da, out);
            walk(b, db, out);
        }

        // A comparison's derivative is identically zero in both operands;
        // every requires-grad parent must still receive a gradient.
        ScalarKind::Cmp { a, b, .. } => {
            walk(a, zero_like(a), out);
            walk(b, zero_like(b), out);
        }

        ScalarKind::Select { c, t, f } => {
            walk(c, zero_like(c), out);
            let zt = zero_like(t);
            let zf = zero_like(f);
            walk(t, select(c.clone(), seed.clone(), zt), out);
            walk(f, select(c.clone(), zf, seed), out);
        }

        ScalarKind::Cast { x, .. } => walk(x, ScalarExpr::cast(x.dtype(), seed), out),
        ScalarKind::Bitcast { x, .. } => walk(x, ScalarExpr::bitcast(x.dtype(), seed), out),

        // Derivative is 0 almost everywhere. QAT straight-through comes from
        // the backward `fake_quant` registers on the sugar node, not from here.
        ScalarKind::Round { x, .. } => walk(x, zero_like(x), out),

        ScalarKind::Dot { a, b } => {
            walk(a, mul(seed.clone(), b.clone()), out);
            walk(b, mul(seed, a.clone()), out);
        }

        // A splat broadcasts one lane into `lanes`; its adjoint sums the
        // incoming lanes back down.
        ScalarKind::Splat { lanes, x } => {
            let ones = ScalarExpr::new(
                ScalarKind::Splat {
                    lanes: *lanes,
                    x: one_of(x.dtype()),
                },
                seed.dtype(),
            );
            let sum = ScalarExpr::new(
                ScalarKind::Dot {
                    a: seed,
                    b: ones,
                },
                x.dtype(),
            );
            walk(x, sum, out);
        }
    }
}

/// `d(op(x))/dx * s`, or `None` where the operand carries no gradient.
fn unary_derivative(op: UnOp, x: &ScalarExpr, s: &ScalarExpr) -> Option<ScalarExpr> {
    let dt = x.dtype();
    let un = |o: UnOp| ScalarExpr::un(o, x.clone());
    let k = |v: f32| lit_of(dt, v);
    Some(match op {
        UnOp::Exp => mul(s.clone(), un(UnOp::Exp)),
        // The adjoint of an approximate exponential is that same approximation.
        UnOp::ApproximateExp => mul(s.clone(), un(UnOp::ApproximateExp)),
        UnOp::LessApproximateExp => mul(s.clone(), un(UnOp::LessApproximateExp)),
        UnOp::Exp2 => mul(mul(s.clone(), un(UnOp::Exp2)), k(std::f32::consts::LN_2)),
        UnOp::Log => div(s.clone(), x.clone()),
        UnOp::Log2 => div(s.clone(), mul(x.clone(), k(std::f32::consts::LN_2))),
        UnOp::Sqrt => div(s.clone(), mul(k(2.0), un(UnOp::Sqrt))),
        UnOp::InverseSqrt => {
            let r = un(UnOp::InverseSqrt);
            mul(mul(k(-0.5), s.clone()), mul(r.clone(), mul(r.clone(), r)))
        }
        UnOp::Sin => mul(s.clone(), un(UnOp::Cos)),
        UnOp::Cos => neg(mul(s.clone(), un(UnOp::Sin))),
        UnOp::Tan => {
            let c = un(UnOp::Cos);
            div(s.clone(), mul(c.clone(), c))
        }
        UnOp::Tanh => {
            let t = un(UnOp::Tanh);
            mul(s.clone(), sub(k(1.0), mul(t.clone(), t)))
        }
        UnOp::Asin => div(s.clone(), sqrt(sub(k(1.0), sqr(x.clone())))),
        UnOp::Acos => neg(div(s.clone(), sqrt(sub(k(1.0), sqr(x.clone()))))),
        UnOp::Atan => div(s.clone(), add(k(1.0), sqr(x.clone()))),
        UnOp::Sinh => mul(s.clone(), un(UnOp::Cosh)),
        UnOp::Cosh => mul(s.clone(), un(UnOp::Sinh)),
        UnOp::Asinh => div(s.clone(), sqrt(add(sqr(x.clone()), k(1.0)))),
        UnOp::Acosh => div(
            s.clone(),
            mul(
                sqrt(sub(x.clone(), k(1.0))),
                sqrt(add(x.clone(), k(1.0))),
            ),
        ),
        UnOp::Atanh => div(s.clone(), sub(k(1.0), sqr(x.clone()))),
        UnOp::Abs => {
            let sign = sub(
                ScalarExpr::cmp(CmpOp::Gt, x.clone(), k(0.0)),
                ScalarExpr::cmp(CmpOp::Lt, x.clone(), k(0.0)),
            );
            mul(s.clone(), sign)
        }
        UnOp::Neg => neg(s.clone()),
        UnOp::Unpack2x16Float => return None,
    })
}

/// `(d/da, d/db)` scaled by `s`.
fn binary_derivative(
    op: BinOp,
    a: &ScalarExpr,
    b: &ScalarExpr,
    s: &ScalarExpr,
) -> (ScalarExpr, ScalarExpr) {
    let dt = a.dtype();
    let k = |v: f32| lit_of(dt, v);
    match op {
        BinOp::Add => (s.clone(), s.clone()),
        BinOp::Sub => (s.clone(), neg(s.clone())),
        BinOp::Mul => (mul(s.clone(), b.clone()), mul(s.clone(), a.clone())),
        BinOp::Div => (
            div(s.clone(), b.clone()),
            neg(div(mul(s.clone(), a.clone()), mul(b.clone(), b.clone()))),
        ),
        BinOp::Pow => (
            mul(
                mul(s.clone(), b.clone()),
                ScalarExpr::bin(BinOp::Pow, a.clone(), sub(b.clone(), k(1.0))),
            ),
            mul(
                mul(
                    s.clone(),
                    ScalarExpr::bin(BinOp::Pow, a.clone(), b.clone()),
                ),
                ScalarExpr::un(UnOp::Log, a.clone()),
            ),
        ),
        // Each side gets a strict mask, so a tie sends the gradient nowhere.
        //
        // `TiePolicy::FirstWins` is not this: it belongs to `Combine::Max`, the
        // fold, where exactly one element must own the reduction.
        BinOp::Max => (
            select(ScalarExpr::cmp(CmpOp::Gt, a.clone(), b.clone()), s.clone(), k(0.0)),
            select(ScalarExpr::cmp(CmpOp::Gt, b.clone(), a.clone()), s.clone(), k(0.0)),
        ),
        BinOp::Min => (
            select(ScalarExpr::cmp(CmpOp::Lt, a.clone(), b.clone()), s.clone(), k(0.0)),
            select(ScalarExpr::cmp(CmpOp::Lt, b.clone(), a.clone()), s.clone(), k(0.0)),
        ),
        // `a - floor(a/b)*b`: the b-partial is a.e. `-trunc(a/b)`, but `Rem`
        // is only reachable on integers at Logical, where no gradient flows.
        BinOp::Rem => (s.clone(), lit_of(b.dtype(), 0.0)),
        BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::BitXor
        | BinOp::Shr
        | BinOp::Shl
        | BinOp::LogicalAnd
        | BinOp::LogicalOr => (lit_of(dt, 0.0), lit_of(b.dtype(), 0.0)),
    }
}

// Every builder folds the literal identities it can prove, which keeps a
// comparison's partial a single `Lit(0)` so `map_adjoint` emits a zero
// `Const` tensor rather than a kernel.

fn lit_of(dtype: Dtype, v: f32) -> ScalarExpr {
    let splat = match dtype {
        Dtype::F32 => Splat::F32(v),
        Dtype::F16 => Splat::F16(half::f16::from_f32(v).to_bits()),
        Dtype::BF16 => Splat::BF16(half::bf16::from_f32(v).to_bits()),
        Dtype::U32 => Splat::U32(v.max(0.0) as u32),
        // Quantized values never reach a scalar body; fall back to a bit
        // pattern rather than panicking inside a pure function.
        Dtype::I32 | Dtype::Q(_) => Splat::I32(v as i32),
    };
    ScalarExpr::lit(splat)
}

fn zero_like(e: &ScalarExpr) -> ScalarExpr {
    lit_of(e.dtype(), 0.0)
}

fn one_of(dtype: Dtype) -> ScalarExpr {
    lit_of(dtype, 1.0)
}

/// True for a structurally-zero literal.
pub fn is_zero(e: &ScalarExpr) -> bool {
    matches!(e.kind(), ScalarKind::Lit(l) if l.0.bits() == 0)
}

fn is_one(e: &ScalarExpr) -> bool {
    match e.kind() {
        ScalarKind::Lit(l) => l.0 == one_splat(l.0.dtype()),
        _ => false,
    }
}

fn one_splat(dtype: Dtype) -> Splat {
    match dtype {
        Dtype::F32 => Splat::F32(1.0),
        Dtype::F16 => Splat::F16(half::f16::from_f32(1.0).to_bits()),
        Dtype::BF16 => Splat::BF16(half::bf16::from_f32(1.0).to_bits()),
        Dtype::U32 => Splat::U32(1),
        Dtype::I32 | Dtype::Q(_) => Splat::I32(1),
    }
}

fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    if is_zero(&a) {
        return b;
    }
    if is_zero(&b) {
        return a;
    }
    ScalarExpr::bin(BinOp::Add, a, b)
}

fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    if is_zero(&b) {
        return a;
    }
    ScalarExpr::bin(BinOp::Sub, a, b)
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    if is_zero(&a) {
        return a;
    }
    if is_zero(&b) {
        return b;
    }
    if is_one(&a) {
        return b;
    }
    if is_one(&b) {
        return a;
    }
    ScalarExpr::bin(BinOp::Mul, a, b)
}

fn div(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    if is_zero(&a) {
        return a;
    }
    if is_one(&b) {
        return a;
    }
    ScalarExpr::bin(BinOp::Div, a, b)
}

fn neg(a: ScalarExpr) -> ScalarExpr {
    if is_zero(&a) {
        return a;
    }
    ScalarExpr::un(UnOp::Neg, a)
}

fn sqr(a: ScalarExpr) -> ScalarExpr {
    ScalarExpr::bin(BinOp::Mul, a.clone(), a)
}

fn sqrt(a: ScalarExpr) -> ScalarExpr {
    ScalarExpr::un(UnOp::Sqrt, a)
}

fn select(c: ScalarExpr, t: ScalarExpr, f: ScalarExpr) -> ScalarExpr {
    if is_zero(&t) && is_zero(&f) {
        return t;
    }
    ScalarExpr::select(c, t, f)
}

/// `d(expr)/d(Arg(arg))` with a unit seed, as a `ScalarExpr` over the same
/// argument list. Comparisons differentiate to a literal zero, which is what
/// makes "every requires-grad parent receives a gradient" hold without a
/// special case.
pub fn d_expr(expr: &ScalarExpr, arg: u32) -> Result<ScalarExpr> {
    let nargs = max_arg(expr).map_or(arg + 1, |m| m.max(arg) + 1);
    let seed = one_of(expr.dtype());
    let partials = differentiate(expr, &seed, nargs);
    Ok(partials
        .get(arg as usize)
        .and_then(|p| p.clone())
        .unwrap_or_else(|| lit_of(expr.dtype(), 0.0)))
}

/// Highest `Arg` index appearing in `expr`.
pub fn max_arg(expr: &ScalarExpr) -> Option<u32> {
    match expr.kind() {
        ScalarKind::Arg(i) => Some(*i),
        ScalarKind::Lit(_) | ScalarKind::Uniform(_) | ScalarKind::IndexOf(_) => None,
        ScalarKind::Un { x, .. }
        | ScalarKind::Cast { x, .. }
        | ScalarKind::Bitcast { x, .. }
        | ScalarKind::Round { x, .. }
        | ScalarKind::Splat { x, .. } => max_arg(x),
        ScalarKind::Bin { a, b, .. } | ScalarKind::Cmp { a, b, .. } | ScalarKind::Dot { a, b } => {
            match (max_arg(a), max_arg(b)) {
                (Some(x), Some(y)) => Some(x.max(y)),
                (v, None) | (None, v) => v,
            }
        }
        ScalarKind::Select { c, t, f } => [max_arg(c), max_arg(t), max_arg(f)]
            .into_iter()
            .flatten()
            .max(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a0() -> ScalarExpr {
        ScalarExpr::arg(0, Dtype::F32)
    }
    fn a1() -> ScalarExpr {
        ScalarExpr::arg(1, Dtype::F32)
    }
    fn f(v: f32) -> ScalarExpr {
        lit_of(Dtype::F32, v)
    }
    fn s() -> ScalarExpr {
        ScalarExpr::arg(9, Dtype::F32)
    }

    /// Numeric evaluation of a partial, for finite-difference checks.
    fn eval(e: &ScalarExpr, args: &[f32]) -> f32 {
        match e.kind() {
            ScalarKind::Arg(i) => args[*i as usize],
            ScalarKind::Lit(l) => match l.0 {
                Splat::F32(v) => v,
                Splat::F16(b) => half::f16::from_bits(b).to_f32(),
                Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
                Splat::U32(v) => v as f32,
                Splat::I32(v) => v as f32,
            },
            ScalarKind::Uniform(_) | ScalarKind::IndexOf(_) => 0.0,
            ScalarKind::Un { op, x } => {
                let v = eval(x, args);
                match op {
                    UnOp::Exp
                    | UnOp::ApproximateExp
                    | UnOp::LessApproximateExp => v.exp(),
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
                    UnOp::Unpack2x16Float => v,
                }
            }
            ScalarKind::Bin { op, a, b } => {
                let (x, y) = (eval(a, args), eval(b, args));
                match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    BinOp::Div => x / y,
                    BinOp::Rem => x % y,
                    BinOp::Pow => x.powf(y),
                    BinOp::Min => x.min(y),
                    BinOp::Max => x.max(y),
                    _ => 0.0,
                }
            }
            ScalarKind::Cmp { op, a, b } => {
                let (x, y) = (eval(a, args), eval(b, args));
                let t = match op {
                    CmpOp::Lt => x < y,
                    CmpOp::Le => x <= y,
                    CmpOp::Gt => x > y,
                    CmpOp::Ge => x >= y,
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                };
                if t { 1.0 } else { 0.0 }
            }
            ScalarKind::Select { c, t, f } => {
                if eval(c, args) != 0.0 {
                    eval(t, args)
                } else {
                    eval(f, args)
                }
            }
            ScalarKind::Cast { x, .. } | ScalarKind::Bitcast { x, .. } => eval(x, args),
            ScalarKind::Round { x, .. } => eval(x, args).round(),
            ScalarKind::Dot { a, b } => eval(a, args) * eval(b, args),
            ScalarKind::Splat { x, .. } => eval(x, args),
        }
    }

    /// Central difference of `expr` in slot `slot` at `args`.
    fn fd(expr: &ScalarExpr, args: &[f32], slot: usize) -> f32 {
        const H: f32 = 1e-3;
        let mut lo = args.to_vec();
        let mut hi = args.to_vec();
        lo[slot] -= H;
        hi[slot] += H;
        (eval(expr, &hi) - eval(expr, &lo)) / (2.0 * H)
    }

    fn check_unary(op: UnOp, x: f32) {
        let expr = ScalarExpr::un(op, a0());
        let partials = differentiate(&expr, &f(1.0), 1);
        let got = eval(partials[0].as_ref().unwrap(), &[x]);
        let want = fd(&expr, &[x], 0);
        let tol = 2e-3 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "{op:?} at {x}: analytic {got} vs finite difference {want}"
        );
    }

    #[test]
    fn every_unary_matches_central_differences_inside_its_domain() {
        for (op, x) in [
            (UnOp::Exp, 0.7),
            (UnOp::Exp2, 0.7),
            (UnOp::Log, 1.7),
            (UnOp::Log2, 1.7),
            (UnOp::Sqrt, 2.3),
            (UnOp::InverseSqrt, 2.3),
            (UnOp::Sin, 0.6),
            (UnOp::Cos, 0.6),
            (UnOp::Tan, 0.4),
            (UnOp::Tanh, 0.5),
            (UnOp::Asin, 0.4),
            (UnOp::Acos, 0.4),
            (UnOp::Atan, 0.9),
            (UnOp::Sinh, 0.6),
            (UnOp::Cosh, 0.6),
            (UnOp::Asinh, 0.8),
            (UnOp::Acosh, 2.1),
            (UnOp::Atanh, 0.3),
            (UnOp::Abs, 1.4),
            (UnOp::Abs, -1.4),
            (UnOp::Neg, 1.1),
        ] {
            check_unary(op, x);
        }
    }

    #[test]
    fn unpack2x16float_gives_a_zero_partial() {
        let expr = ScalarExpr::un(UnOp::Unpack2x16Float, a0());
        let partials = differentiate(&expr, &f(1.0), 1);
        assert!(is_zero(partials[0].as_ref().unwrap()));
    }

    #[test]
    fn every_binary_matches_central_differences() {
        for (op, x, y) in [
            (BinOp::Add, 1.3, 0.7),
            (BinOp::Sub, 1.3, 0.7),
            (BinOp::Mul, 1.3, 0.7),
            (BinOp::Div, 1.3, 0.7),
            (BinOp::Pow, 1.3, 2.4),
            (BinOp::Max, 1.3, 0.7),
            (BinOp::Min, 1.3, 0.7),
        ] {
            let expr = ScalarExpr::bin(op, a0(), a1());
            let partials = differentiate(&expr, &f(1.0), 2);
            for slot in 0..2 {
                let got = eval(partials[slot].as_ref().unwrap(), &[x, y]);
                let want = fd(&expr, &[x, y], slot);
                let tol = 3e-3 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tol,
                    "{op:?} slot {slot}: {got} vs {want}"
                );
            }
        }
    }

    /// A tie sends the gradient to neither side.
    #[test]
    fn max_and_min_send_a_tie_to_neither_side() {
        for op in [BinOp::Max, BinOp::Min] {
            let expr = ScalarExpr::bin(op, a0(), a1());
            let p = differentiate(&expr, &f(1.0), 2);
            assert_eq!(eval(p[0].as_ref().unwrap(), &[2.0, 2.0]), 0.0, "{op:?} lhs");
            assert_eq!(eval(p[1].as_ref().unwrap(), &[2.0, 2.0]), 0.0, "{op:?} rhs");
        }
        // Away from the tie the winner still takes the whole gradient.
        let mx = ScalarExpr::bin(BinOp::Max, a0(), a1());
        let p = differentiate(&mx, &f(1.0), 2);
        assert_eq!(eval(p[0].as_ref().unwrap(), &[3.0, 2.0]), 1.0);
        assert_eq!(eval(p[1].as_ref().unwrap(), &[3.0, 2.0]), 0.0);
        let mn = ScalarExpr::bin(BinOp::Min, a0(), a1());
        let p = differentiate(&mn, &f(1.0), 2);
        assert_eq!(eval(p[0].as_ref().unwrap(), &[3.0, 2.0]), 0.0);
        assert_eq!(eval(p[1].as_ref().unwrap(), &[3.0, 2.0]), 1.0);
    }

    #[test]
    fn all_twelve_comparisons_differentiate_to_a_literal_zero() {
        for op in [
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge,
            CmpOp::Eq,
            CmpOp::Ne,
        ] {
            // tensor-tensor
            let tt = ScalarExpr::cmp(op, a0(), a1());
            for p in differentiate(&tt, &f(1.0), 2) {
                assert!(is_zero(&p.expect("a comparison still yields a partial")));
            }
            // tensor-scalar
            let ts = ScalarExpr::cmp(op, a0(), f(0.5));
            for p in differentiate(&ts, &f(1.0), 1) {
                assert!(is_zero(&p.expect("a comparison still yields a partial")));
            }
        }
    }

    #[test]
    fn where_cond_routes_the_gradient_through_the_mask() {
        let expr = ScalarExpr::select(a0(), a1(), ScalarExpr::arg(2, Dtype::F32));
        let p = differentiate(&expr, &s(), 3);
        let mut args = vec![0.0; 10];
        args[9] = 4.0; // the seed
        args[0] = 1.0; // condition true
        assert!(is_zero(p[0].as_ref().unwrap()), "the condition gets zeros");
        assert_eq!(eval(p[1].as_ref().unwrap(), &args), 4.0);
        assert_eq!(eval(p[2].as_ref().unwrap(), &args), 0.0);
        args[0] = 0.0; // condition false
        assert_eq!(eval(p[1].as_ref().unwrap(), &args), 0.0);
        assert_eq!(eval(p[2].as_ref().unwrap(), &args), 4.0);
    }

    #[test]
    fn clamp_differentiates_to_the_two_sided_mask() {
        // clamp(x, lo, hi) = min(max(x, lo), hi)
        let expr = ScalarExpr::bin(
            BinOp::Min,
            ScalarExpr::bin(BinOp::Max, a0(), f(-1.0)),
            f(2.0),
        );
        let p = differentiate(&expr, &f(1.0), 1);
        let d = p[0].as_ref().unwrap();
        assert_eq!(eval(d, &[0.5]), 1.0, "inside the band the gradient passes");
        assert_eq!(eval(d, &[-3.0]), 0.0, "below `lo` it is clipped");
        assert_eq!(eval(d, &[9.0]), 0.0, "above `hi` it is clipped");
    }

    #[test]
    fn cast_routes_the_gradient_back_to_the_source_dtype() {
        let x = ScalarExpr::arg(0, Dtype::F32);
        let expr = ScalarExpr::cast(Dtype::F16, x);
        let seed = ScalarExpr::arg(1, Dtype::F16);
        let p = differentiate(&expr, &seed, 1);
        let d = p[0].as_ref().unwrap();
        assert_eq!(d.dtype(), Dtype::F32, "the master weight keeps f32");
        assert!(matches!(d.kind(), ScalarKind::Cast { to: Dtype::F32, .. }));
    }

    #[test]
    fn round_has_a_zero_derivative() {
        let expr = ScalarExpr::round(fusor2_ir::dtype::RoundMode::HalfAwayFromZero, a0());
        let p = differentiate(&expr, &f(1.0), 1);
        assert!(is_zero(p[0].as_ref().unwrap()));
    }

    #[test]
    fn relu_sigmoid_silu_and_tanh_exact_fall_out_of_the_table() {
        // relu = max(x, 0)
        let relu = ScalarExpr::bin(BinOp::Max, a0(), f(0.0));
        let d = differentiate(&relu, &f(1.0), 1)[0].clone().unwrap();
        assert_eq!(eval(&d, &[1.5]), 1.0);
        assert_eq!(eval(&d, &[-1.5]), 0.0);

        // sigmoid = 1 / (1 + exp(-x)); d = sigmoid * (1 - sigmoid)
        let sig = ScalarExpr::bin(
            BinOp::Div,
            f(1.0),
            ScalarExpr::bin(
                BinOp::Add,
                f(1.0),
                ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, a0())),
            ),
        );
        let d = differentiate(&sig, &f(1.0), 1)[0].clone().unwrap();
        let s = 1.0 / (1.0 + (-0.3f32).exp());
        assert!((eval(&d, &[0.3]) - s * (1.0 - s)).abs() < 1e-4);

        // silu = x * sigmoid(x)
        let silu = ScalarExpr::bin(BinOp::Mul, a0(), sig.clone());
        let d = differentiate(&silu, &f(1.0), 1)[0].clone().unwrap();
        assert!((eval(&d, &[0.3]) - fd(&silu, &[0.3], 0)).abs() < 2e-3);

        // tanh_exact = (e^x - e^-x) / (e^x + e^-x)
        let ex = ScalarExpr::un(UnOp::Exp, a0());
        let enx = ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, a0()));
        let te = ScalarExpr::bin(
            BinOp::Div,
            ScalarExpr::bin(BinOp::Sub, ex.clone(), enx.clone()),
            ScalarExpr::bin(BinOp::Add, ex, enx),
        );
        let d = differentiate(&te, &f(1.0), 1)[0].clone().unwrap();
        assert!((eval(&d, &[0.4]) - (1.0 - 0.4f32.tanh().powi(2))).abs() < 2e-3);
    }

    /// `0.5*x*(1 + tanh(c*(x + 0.044715*x^3)))` differentiates to
    /// `0.5*(1+t) + 0.5*x*(1-t^2)*c*(1 + 3*0.044715*x^2)`.
    #[test]
    fn gelu_differentiates_to_its_published_analytic_form() {
        const C: f32 = 0.797_884_6; // sqrt(2/pi)
        const K: f32 = 0.044_715;
        let x = a0();
        let inner = ScalarExpr::bin(
            BinOp::Add,
            x.clone(),
            ScalarExpr::bin(
                BinOp::Mul,
                f(K),
                ScalarExpr::bin(BinOp::Mul, x.clone(), ScalarExpr::bin(BinOp::Mul, x.clone(), x.clone())),
            ),
        );
        let t = ScalarExpr::un(UnOp::Tanh, ScalarExpr::bin(BinOp::Mul, f(C), inner));
        let gelu = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::bin(BinOp::Mul, f(0.5), x.clone()),
            ScalarExpr::bin(BinOp::Add, f(1.0), t.clone()),
        );
        let d = differentiate(&gelu, &f(1.0), 1)[0].clone().unwrap();
        for probe in [-2.0f32, -0.5, 0.0, 0.5, 2.0] {
            let tv = (C * (probe + K * probe * probe * probe)).tanh();
            let want = 0.5 * (1.0 + tv)
                + 0.5 * probe * (1.0 - tv * tv) * C * (1.0 + 3.0 * K * probe * probe);
            let got = eval(&d, &[probe]);
            assert!(
                (got - want).abs() <= 2e-3 * want.abs().max(1.0),
                "gelu' at {probe}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn a_repeated_arg_accumulates_both_paths() {
        // x*x: both operands are Arg(0), so the partial must be 2x.
        let expr = ScalarExpr::bin(BinOp::Mul, a0(), a0());
        let d = differentiate(&expr, &f(1.0), 1)[0].clone().unwrap();
        assert!((eval(&d, &[3.0]) - 6.0).abs() < 1e-5);
    }

    #[test]
    fn d_expr_agrees_with_differentiate() {
        let expr = ScalarExpr::bin(BinOp::Mul, a0(), a1());
        let d = d_expr(&expr, 1).unwrap();
        assert_eq!(eval(&d, &[2.0, 5.0]), 2.0);
    }
}
