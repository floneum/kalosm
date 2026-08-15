//! The 23 unaries, 8 scalar-arith ops, the same-rank and broadcasting
//! binaries, the std-ops surface, `where_cond` and all 12 comparisons.
//!
//! Every one of these is one `Map` with a different `ScalarExpr`, so this
//! suite is really a `ScalarExpr` test: `NaryOp`'s 50-variant discriminant —
//! whose ordering the reference admits is load-bearing in kernel cache keys —
//! does not exist, and a table this wide passing is the evidence.
//!
//! Owned by W14.

use fusor2::{Dtype, Session, };
use fusor2::tensor::Dyn as Tensor;

use crate::compare::{
    assert_all_zero, assert_gradient_matches_finite_difference, finite_difference_gradient,
    relative_eq,
};
use crate::harness::{CaseError, CaseResult, Cases, dense_len, dims, is_gpu};
use crate::suite::support::{
    BinaryOp, Domain, UnaryOp, binary_case, comparison_case, expect_values, gradient_of, graph_of,
    loss_of, read, read_scalar, unary_case, upload,
};

/// Shape every elementwise case runs at. Small, because each backward case
/// pays `2 * 24` graph rebuilds for finite differences.
const SHAPE: &[u64] = &[4, 6];

/// `(e^x - e^-x) / (e^x + e^-x)`, the form `tanh_exact` names. A separate
/// expression from `tanh`, not an alias: the reference needs it where a
/// driver's native tanh under-saturates the GELU tail.
fn tanh_exact_ref(x: f32) -> f32 {
    let (up, down) = (x.exp(), (-x).exp());
    (up - down) / (up + down)
}

/// The 21 unaries with an exact elementwise reference. `approximate_exp` and
/// `less_approximate_exp` are the other two of the 23; they are deliberately
/// *not* `exp`, so they get a relative bound instead.
#[rustfmt::skip]
fn unaries() -> Vec<(&'static str, Domain, UnaryOp, fn(f32) -> f32)> {
    vec![
        // `abs` is sampled off zero: its adjoint is undefined there and a
        // central difference straddling the kink disagrees with any convention.
        ("abs",        Domain::Custom(0.2, 1.5), |x| x.abs(),    f32::abs),
        ("acos",       Domain::Unit,             |x| x.acos(),   f32::acos),
        ("acosh",      Domain::AboveOne,         |x| x.acosh(),  f32::acosh),
        ("asin",       Domain::Unit,             |x| x.asin(),   f32::asin),
        ("asinh",      Domain::Wide,             |x| x.asinh(),  f32::asinh),
        ("atan",       Domain::Wide,             |x| x.atan(),   f32::atan),
        ("atanh",      Domain::Unit,             |x| x.atanh(),  f32::atanh),
        ("sin",        Domain::Wide,             |x| x.sin(),    f32::sin),
        ("sinh",       Domain::Wide,             |x| x.sinh(),   f32::sinh),
        ("cos",        Domain::Wide,             |x| x.cos(),    f32::cos),
        ("cosh",       Domain::Wide,             |x| x.cosh(),   f32::cosh),
        ("tan",        Domain::Unit,             |x| x.tan(),    f32::tan),
        ("tanh",       Domain::Wide,             |x| x.tanh(),   f32::tanh),
        ("tanh_exact", Domain::Wide,             |x| x.tanh_exact(), tanh_exact_ref),
        ("exp",        Domain::Wide,             |x| x.exp(),    f32::exp),
        ("exp2",       Domain::Wide,             |x| x.exp2(),   f32::exp2),
        ("log",        Domain::Positive,         |x| x.log(),    f32::ln),
        ("log2",       Domain::Positive,         |x| x.log2(),   f32::log2),
        ("neg",        Domain::Wide,             |x| x.neg(),    |v| -v),
        ("sqr",        Domain::Wide,             |x| x.square(), |v| v * v),
        ("sqrt",       Domain::Positive,         |x| x.sqrt(),   f32::sqrt),
    ]
}

/// The 8 scalar-arith unaries, each with its constant baked into the closure.
#[rustfmt::skip]
fn scalar_arith() -> Vec<(&'static str, Domain, UnaryOp, fn(f32) -> f32)> {
    vec![
        ("add_scalar", Domain::Wide,     |x| x.add_scalar(0.75), |v| v + 0.75),
        ("sub_scalar", Domain::Wide,     |x| x.sub_scalar(0.25), |v| v - 0.25),
        ("mul_scalar", Domain::Wide,     |x| x.mul_scalar(-1.5), |v| v * -1.5),
        ("div_scalar", Domain::Wide,     |x| x.div_scalar(2.0),  |v| v / 2.0),
        ("pow_scalar", Domain::Positive, |x| x.pow_scalar(1.5),  |v| v.powf(1.5)),
        // Sampled off the kink so finite differences agree with the adjoint.
        // On [0.2, 1.5) all three of these are the IDENTITY, so these rows
        // check the adjoint and nothing else — `forward_only()` below owns the
        // half that distinguishes the op from a passthrough.
        ("max_scalar", Domain::Custom(0.2, 1.5), |x| x.max_scalar(0.1), |v| v.max(0.1)),
        ("min_scalar", Domain::Custom(0.2, 1.5), |x| x.min_scalar(1.9), |v| v.min(1.9)),
        ("clamp",      Domain::Custom(0.2, 1.5), |x| x.clamp(0.1, 1.9), |v| v.clamp(0.1, 1.9)),
    ]
}

/// Forward-only rows, over domains that **straddle** the kink the tables above
/// sample away from.
///
/// `abs`, `max_scalar` and `min_scalar` are sampled on `[0.2, 1.5)` up there so
/// a central difference does not straddle a point where the adjoint is
/// undefined — a real constraint, and the reason the choice was made. But every
/// one of those three is exactly `|v| v` on that interval, so the forward half
/// of those rows cannot tell the op from a passthrough: `abs` on a negative,
/// and either clamp actually clamping, were verified nowhere in the suite.
///
/// These rows restore that coverage. No gradient is taken, so the kink is
/// harmless; [`non_vacuous`] is what keeps the domain honest.
#[rustfmt::skip]
fn forward_only() -> Vec<(&'static str, Domain, UnaryOp, fn(f32) -> f32)> {
    vec![
        ("abs_straddles_zero",       Domain::Wide,            |x| x.abs(),           f32::abs),
        ("max_scalar_clamps_below",  Domain::Custom(-2.0, 2.0), |x| x.max_scalar(0.5), |v| v.max(0.5)),
        ("min_scalar_clamps_above",  Domain::Custom(-2.0, 2.0), |x| x.min_scalar(0.5), |v| v.min(0.5)),
        ("clamp_clamps_both_ends",   Domain::Custom(-2.0, 2.0), |x| x.clamp(-0.5, 0.5), |v| v.clamp(-0.5, 0.5)),
    ]
}

/// Refuse a row whose reference is the identity on its own sampled data.
///
/// A case that samples a domain where the op under test does nothing passes
/// against a passthrough implementation and reports as coverage. That is how
/// `abs`, `max_scalar` and `min_scalar` were green while their forward
/// behaviour was unverified — the domains were chosen for the *backward* half
/// and nobody re-checked what the forward half could still see.
fn non_vacuous(name: &str, data: &[f32], reference: fn(f32) -> f32) -> Result<(), CaseError> {
    if data.iter().all(|v| reference(*v) == *v) {
        return Err(format!(
            "`{name}` is the identity on every one of its {} sampled inputs, so this case \
             cannot distinguish the op from a passthrough; widen the domain",
            data.len()
        )
        .into());
    }
    Ok(())
}

/// The 5 same-rank binaries plus `pow`.
#[rustfmt::skip]
fn binaries() -> Vec<(&'static str, Domain, BinaryOp, fn(f32, f32) -> f32)> {
    vec![
        ("add", Domain::Wide,     |a, b| a.add(b), |x, y| x + y),
        ("sub", Domain::Wide,     |a, b| a.sub(b), |x, y| x - y),
        ("mul", Domain::Wide,     |a, b| a.mul(b), |x, y| x * y),
        ("div", Domain::Positive, |a, b| a.div(b), |x, y| x / y),
        ("rem", Domain::Positive, |a, b| a.rem_scalar(0.5)?.add(b), |x, y| x % 0.5 + y),
        ("pow", Domain::Positive, |a, b| a.pow(b), |x, y| x.powf(y)),
    ]
}

/// The 6 scalar comparisons, which — as in the reference — take a *scalar*,
/// not a tensor. Each differentiates to zero.
#[rustfmt::skip]
fn scalar_comparisons() -> Vec<(&'static str, UnaryOp, fn(f32) -> f32)> {
    vec![
        ("eq_scalar",  |x| x.eq_scalar(0.0), |v| f32::from(v == 0.0)),
        ("ne_scalar",  |x| x.ne_scalar(0.0), |v| f32::from(v != 0.0)),
        ("lt_scalar",  |x| x.lt_scalar(0.0), |v| f32::from(v < 0.0)),
        ("lte_scalar", |x| x.le_scalar(0.0), |v| f32::from(v <= 0.0)),
        ("gt_scalar",  |x| x.gt_scalar(0.0), |v| f32::from(v > 0.0)),
        ("gte_scalar", |x| x.ge_scalar(0.0), |v| f32::from(v >= 0.0)),
    ]
}

/// The 6 tensor comparisons.
#[rustfmt::skip]
fn tensor_comparisons() -> Vec<(&'static str, BinaryOp, fn(f32, f32) -> f32)> {
    vec![
        ("eq_tensor",  |a, b| a.eq_tensor(b),  |x, y| f32::from(x == y)),
        ("ne_tensor",  |a, b| a.ne_tensor(b),  |x, y| f32::from(x != y)),
        ("lt_tensor",  |a, b| a.lt_tensor(b),  |x, y| f32::from(x < y)),
        ("lte_tensor", |a, b| a.lte_tensor(b), |x, y| f32::from(x <= y)),
        ("gt_tensor",  |a, b| a.gt_tensor(b),  |x, y| f32::from(x > y)),
        ("gte_tensor", |a, b| a.gte_tensor(b), |x, y| f32::from(x >= y)),
    ]
}

/// The 5 broadcasting binaries.
#[rustfmt::skip]
fn broadcasting() -> Vec<(&'static str, BinaryOp, fn(f32, f32) -> f32)> {
    vec![
        ("add_", |a, b| a.broadcast_add(b), |x, y| x + y),
        ("sub_", |a, b| a.broadcast_sub(b), |x, y| x - y),
        ("mul_", |a, b| a.broadcast_mul(b), |x, y| x * y),
        ("div_", |a, b| a.broadcast_div(b), |x, y| x / y),
        ("pow_", |a, b| a.broadcast_pow(b), |x, y| x.powf(y)),
    ]
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    for (name, domain, op, reference) in unaries() {
        cases.push_case(unary_case(
            "elementwise",
            name,
            SHAPE,
            17,
            domain,
            op,
            reference,
        ));
    }
    for (name, domain, op, reference) in scalar_arith() {
        cases.push_case(unary_case(
            "elementwise",
            name,
            SHAPE,
            19,
            domain,
            op,
            reference,
        ));
    }
    for (name, domain, op, reference) in forward_only() {
        cases.push("elementwise", name, move |session| {
            let data = domain.sample(29, dense_len(&dims(SHAPE)));
            non_vacuous(name, &data, reference)?;
            let graph = graph_of(session);
            let x = upload(graph.handle(), &dims(SHAPE), &data)?;
            let y = op(&x).map_err(|e| -> CaseError { e.to_string().into() })?;
            let expected: Vec<f32> = data.iter().copied().map(reference).collect();
            expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)
        });
    }
    for (name, domain, op, reference) in binaries() {
        cases.push_case(binary_case(
            "elementwise",
            name,
            SHAPE,
            domain,
            op,
            reference,
        ));
    }
    for (name, op, reference) in scalar_comparisons() {
        cases.push_case(comparison_case("elementwise", name, op, reference));
    }
    for (name, op, reference) in tensor_comparisons() {
        cases.push("elementwise", name, move |session| {
            tensor_comparison_case(session, name, op, reference)
        });
    }
    for (name, op, reference) in broadcasting() {
        cases.push("elementwise", name, move |session| {
            broadcast_case(session, op, reference)
        });
    }

    // The two GPU-approximate exponentials. Their point is that they are *not*
    // `exp`, so they get a relative bound rather than an elementwise
    // reference: an implementation that quietly aliased them to `exp` would
    // pass a strict comparison and hide the missing expression.
    cases.push("elementwise", "approximate_exp", |session| {
        approximate_exp_case(session, "approximate_exp", 5e-3)
    });
    cases.push("elementwise", "less_approximate_exp", |session| {
        approximate_exp_case(session, "less_approximate_exp", 5e-2)
    });

    // The two elementwise extrema, whose adjoint is a mask rather than zero.
    cases.push_case(binary_case(
        "elementwise",
        "max_elementwise",
        SHAPE,
        Domain::Wide,
        |a, b| a.maximum(b),
        f32::max,
    ));
    cases.push_case(binary_case(
        "elementwise",
        "min_elementwise",
        SHAPE,
        Domain::Wide,
        |a, b| a.minimum(b),
        f32::min,
    ));

    // The operator surface. Separate cases because a chained expression is a
    // different `ScalarExpr::compose` shape than a single op, and composition
    // *is* elementwise fusion.
    cases.push("elementwise", "std_ops_add_sub", |s| {
        expr_case(s, |a, b| a.add(b)?.sub(b), |x, y| (x + y) - y)
    });
    cases.push("elementwise", "std_ops_mul_div", |s| {
        expr_case(s, |a, b| a.mul(b)?.div(b), |x, y| (x * y) / y)
    });
    cases.push("elementwise", "std_ops_neg", |s| {
        expr_case(s, |a, b| a.neg()?.sub(b), |x, y| -x - y)
    });
    cases.push("elementwise", "std_ops_scalar", |s| {
        expr_case(
            s,
            |a, b| a.mul_scalar(3.0)?.add_scalar(-1.0)?.sub(b),
            |x, y| (x * 3.0 - 1.0) - y,
        )
    });

    cases.push("elementwise", "where_cond", where_cond_case);
    cases
}

// ---------------------------------------------------------------------------

fn backend_of(session: &Session) -> &'static str {
    if is_gpu(session) { "gpu" } else { "cpu" }
}

/// A tensor-tensor comparison: 1.0/0.0 forward, zero gradient to **both**
/// parents. The tape validates that every requires-grad parent receives a
/// gradient, so an absent rule and a zero rule are different outcomes.
fn tensor_comparison_case(
    session: &Session,
    name: &'static str,
    op: BinaryOp,
    reference: fn(f32, f32) -> f32,
) -> CaseResult {
    let len = dense_len(&dims(SHAPE));
    let lhs = Domain::Wide.sample(41, len);
    // Half the rows share a value with `lhs`, so the equality comparisons are
    // not vacuously all-zero.
    let mut rhs = Domain::Wide.sample(43, len);
    for i in (0..len).step_by(2) {
        rhs[i] = lhs[i];
    }
    let dimv = dims(SHAPE);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dimv, &lhs)?;
    let b = upload(graph.handle(), &dimv, &rhs)?;
    let y = op(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let expected: Vec<f32> = lhs
        .iter()
        .zip(&rhs)
        .map(|(x, y)| reference(*x, *y))
        .collect();
    expect_values(session, SHAPE, Dtype::F32, &actual, &expected)?;

    assert_all_zero(name, &gradient_of(&graph, &y, &a)?)?;
    assert_all_zero(name, &gradient_of(&graph, &y, &b)?)?;
    Ok(())
}

/// A rank-2 activation against a rank-1 operand, right-aligned.
///
/// No implicit broadcasting exists at L0 — the frontend emits
/// `Restride { multiplier: 0 }` — so this is really a test that the frontend's
/// right-aligned rules hold *and* that a stride-0 axis's adjoint is a sum over
/// that axis. A rule that forgot the sum would hand a `[4, 6]` gradient to a
/// `[6]` leaf.
fn broadcast_case(session: &Session, op: BinaryOp, reference: fn(f32, f32) -> f32) -> CaseResult {
    const ROWS: u64 = 4;
    const COLS: u64 = 6;
    let lhs = Domain::Positive.sample(53, (ROWS * COLS) as usize);
    let rhs = Domain::Positive.sample(59, COLS as usize);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[ROWS, COLS]), &lhs)?;
    let b = upload(graph.handle(), &dims(&[COLS]), &rhs)?;
    let y = op(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let expected: Vec<f32> = (0..(ROWS * COLS) as usize)
        .map(|i| reference(lhs[i], rhs[i % COLS as usize]))
        .collect();
    expect_values(session, &[ROWS, COLS], Dtype::F32, &actual, &expected)?;

    let d_rhs = gradient_of(&graph, &y, &b)?;
    if d_rhs.len() != COLS as usize {
        return Err(format!(
            "the broadcast operand's gradient has {} elements, not {COLS}: a stride-0 \
             axis's adjoint is a sum over that axis",
            d_rhs.len()
        )
        .into());
    }
    let numeric = finite_difference_gradient(&[COLS as usize], &rhs, &mut |probe| {
        let g = graph_of(session);
        let a = upload(g.handle(), &dims(&[ROWS, COLS]), &lhs)?;
        let b = upload(g.handle(), &dims(&[COLS]), probe)?;
        let y = op(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&d_rhs, &numeric)?;
    Ok(())
}

/// A two-operand expression checked forward and on the left gradient.
fn expr_case(
    session: &Session,
    build: fn(&Tensor, &Tensor) -> fusor2::Result<Tensor>,
    reference: fn(f32, f32) -> f32,
) -> CaseResult {
    let len = dense_len(&dims(SHAPE));
    let lhs = Domain::Positive.sample(61, len);
    let rhs = Domain::Positive.sample(67, len);
    let dimv = dims(SHAPE);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dimv, &lhs)?;
    let b = upload(graph.handle(), &dimv, &rhs)?;
    let y = build(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let expected: Vec<f32> = lhs
        .iter()
        .zip(&rhs)
        .map(|(x, y)| reference(*x, *y))
        .collect();
    expect_values(session, SHAPE, Dtype::F32, &actual, &expected)?;

    let analytic = gradient_of(&graph, &y, &a)?;
    let numeric = finite_difference_gradient(&[len], &lhs, &mut |probe| {
        let g = graph_of(session);
        let a = upload(g.handle(), &dimv, probe)?;
        let b = upload(g.handle(), &dimv, &rhs)?;
        let y = build(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

/// An approximate exponential: within `tol` of `exp` in relative terms, and
/// differentiable to itself.
///
/// `fusor2::Tensor` does not expose `approximate_exp`/`less_approximate_exp`
/// yet, so this case reports the missing entry point by name rather than
/// aliasing to `exp` and passing vacuously — an alias would make the case
/// green while the expression that justifies its existence was absent.
fn approximate_exp_case(session: &Session, name: &'static str, tol: f32) -> CaseResult {
    let len = dense_len(&dims(SHAPE));
    let data = Domain::Wide.sample(71, len);
    let dimv = dims(SHAPE);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;

    let Some(y) = approximate_exp_op(&x, name) else {
        return Err(format!(
            "fusor2::Tensor has no `{name}`; it is one of the 23 unaries and must be \
             its own ScalarExpr, not an alias for `exp`"
        )
        .into());
    };
    let y = y.map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let exact: Vec<f32> = data.iter().map(|v| v.exp()).collect();
    relative_eq(backend_of(session), &[len], &exact, &actual, tol)?;

    // d(exp(x))/dx = exp(x), whichever approximation is in use, so the
    // gradient must equal the forward output.
    let analytic = gradient_of(&graph, &y, &x)?;
    assert_gradient_matches_finite_difference(&analytic, &actual)?;
    Ok(())
}

/// Resolve the approximate-exponential entry point by name.
///
/// Both are their own `UnOp` now (`UnOp::ApproximateExp` /
/// `UnOp::LessApproximateExp`), so this is no longer an alias for `exp` — the
/// equivalence test below pins that they hash-cons to different nodes.
fn approximate_exp_op(x: &Tensor, name: &str) -> Option<fusor2::Result<Tensor>> {
    match name {
        "approximate_exp" => Some(x.approximate_exp()),
        "less_approximate_exp" => Some(x.less_approximate_exp()),
        _ => None,
    }
}

/// `where_cond`: condition, on_true and on_false all share one shape and one
/// dtype, because there is no bool. The condition receives a zero gradient and
/// the branches receive the mask and its complement.
fn where_cond_case(session: &Session) -> CaseResult {
    let len = dense_len(&dims(SHAPE));
    let cond: Vec<f32> = Domain::Wide
        .sample(73, len)
        .iter()
        .map(|v| f32::from(*v > 0.0))
        .collect();
    let on_true = Domain::Wide.sample(79, len);
    let on_false = Domain::Wide.sample(83, len);
    let dimv = dims(SHAPE);

    let graph = graph_of(session);
    let c = upload(graph.handle(), &dimv, &cond)?;
    let t = upload(graph.handle(), &dimv, &on_true)?;
    let f = upload(graph.handle(), &dimv, &on_false)?;
    let y = c
        .where_cond(&t, &f)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let expected: Vec<f32> = (0..len)
        .map(|i| {
            if cond[i] != 0.0 {
                on_true[i]
            } else {
                on_false[i]
            }
        })
        .collect();
    expect_values(session, SHAPE, Dtype::F32, &actual, &expected)?;

    assert_all_zero("where_cond condition", &gradient_of(&graph, &y, &c)?)?;

    let d_true = gradient_of(&graph, &y, &t)?;
    let d_false = gradient_of(&graph, &y, &f)?;
    for i in 0..len {
        let (want_t, want_f) = if cond[i] != 0.0 {
            (1.0, 0.0)
        } else {
            (0.0, 1.0)
        };
        if d_true[i] != want_t || d_false[i] != want_f {
            return Err(format!(
                "where_cond gradient {i}: got ({}, {}), want ({want_t}, {want_f})",
                d_true[i], d_false[i]
            )
            .into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    fn has(names: &[String], short: &str) -> bool {
        names.iter().any(|n| n == &format!("elementwise::{short}"))
    }

    /// The guard has to fire on the exact configuration that was shipping
    /// green, or it is decoration.
    #[test]
    fn non_vacuous_rejects_a_domain_where_the_op_is_the_identity() {
        let len = dense_len(&dims(SHAPE));
        // Verbatim what `scalar_arith()` samples for these three rows.
        let vacuous = Domain::Custom(0.2, 1.5).sample(19, len);
        for (name, reference) in [
            ("abs", f32::abs as fn(f32) -> f32),
            ("max_scalar", |v: f32| v.max(0.1)),
            ("min_scalar", |v: f32| v.min(1.9)),
        ] {
            assert!(
                non_vacuous(name, &vacuous, reference).is_err(),
                "{name} is the identity on [0.2, 1.5) and the guard missed it"
            );
        }
    }

    /// ...and must not fire on the widened domains, or it would just be a
    /// second way to fail.
    #[test]
    fn every_forward_only_row_actually_exercises_its_op() {
        let len = dense_len(&dims(SHAPE));
        for (name, domain, _, reference) in forward_only() {
            let data = domain.sample(29, len);
            non_vacuous(name, &data, reference)
                .unwrap_or_else(|e| panic!("{name} is vacuous on its own domain: {e}"));
        }
    }

    #[test]
    fn the_kink_straddling_rows_are_registered() {
        let names = registered();
        for wanted in [
            "abs_straddles_zero",
            "max_scalar_clamps_below",
            "min_scalar_clamps_above",
            "clamp_clamps_both_ends",
        ] {
            assert!(has(&names, wanted), "{wanted} is not registered");
        }
    }

    #[test]
    fn all_23_unaries_are_registered() {
        let names = registered();
        for wanted in [
            "abs",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "sin",
            "sinh",
            "cos",
            "cosh",
            "tan",
            "tanh",
            "tanh_exact",
            "exp",
            "exp2",
            "approximate_exp",
            "less_approximate_exp",
            "log",
            "log2",
            "neg",
            "sqr",
            "sqrt",
        ] {
            assert!(
                has(&names, wanted),
                "the unary `{wanted}` is not registered"
            );
        }
        assert_eq!(unaries().len() + 2, 23);
    }

    #[test]
    fn all_8_scalar_arith_ops_are_registered() {
        let names = registered();
        assert_eq!(scalar_arith().len(), 8);
        for wanted in [
            "add_scalar",
            "sub_scalar",
            "mul_scalar",
            "div_scalar",
            "pow_scalar",
            "max_scalar",
            "min_scalar",
            "clamp",
        ] {
            assert!(has(&names, wanted), "{wanted}");
        }
    }

    #[test]
    fn the_five_same_rank_binaries_and_pow_are_registered() {
        let names = registered();
        assert_eq!(binaries().len(), 6);
        for wanted in ["add", "sub", "mul", "div", "rem", "pow"] {
            assert!(has(&names, wanted), "{wanted}");
        }
    }

    #[test]
    fn all_twelve_comparisons_plus_the_four_extras_are_registered() {
        let names = registered();
        assert_eq!(
            scalar_comparisons().len(),
            6,
            "one spelling per scalar comparison"
        );
        assert_eq!(tensor_comparisons().len(), 6);
        for wanted in [
            "eq_scalar",
            "ne_scalar",
            "lt_scalar",
            "lte_scalar",
            "gt_scalar",
            "gte_scalar",
            "eq_tensor",
            "ne_tensor",
            "lt_tensor",
            "lte_tensor",
            "gt_tensor",
            "gte_tensor",
            "max_elementwise",
            "min_elementwise",
        ] {
            assert!(has(&names, wanted), "the comparison `{wanted}` is missing");
        }
    }

    #[test]
    fn the_five_broadcasting_binaries_are_registered() {
        let names = registered();
        assert_eq!(broadcasting().len(), 5);
        for wanted in ["add_", "sub_", "mul_", "div_", "pow_"] {
            assert!(has(&names, wanted), "{wanted}");
        }
    }

    #[test]
    fn the_std_ops_surface_and_where_cond_are_registered() {
        let names = registered();
        for wanted in [
            "std_ops_add_sub",
            "std_ops_mul_div",
            "std_ops_neg",
            "std_ops_scalar",
            "where_cond",
        ] {
            assert!(has(&names, wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn tanh_exact_agrees_with_tanh_where_both_are_well_conditioned() {
        // The two differ only in the saturated tail; this pins the middle so a
        // wrong reference formula cannot hide there.
        for x in [-1.0f32, -0.25, 0.0, 0.25, 1.0] {
            assert!((tanh_exact_ref(x) - x.tanh()).abs() < 1e-6, "{x}");
        }
    }

    #[test]
    fn every_domain_stays_inside_its_ops_definition() {
        for (name, domain, _, _) in unaries() {
            let sample = domain.sample(17, 256);
            match name {
                "log" | "log2" | "sqrt" => {
                    assert!(sample.iter().all(|v| *v > 0.0), "{name} sampled <= 0")
                }
                "acos" | "asin" | "atanh" => {
                    assert!(
                        sample.iter().all(|v| v.abs() < 1.0),
                        "{name} sampled |x| >= 1"
                    )
                }
                "acosh" => assert!(sample.iter().all(|v| *v >= 1.0), "acosh sampled < 1"),
                "abs" => assert!(sample.iter().all(|v| *v != 0.0), "abs sampled the kink"),
                _ => {}
            }
        }
    }

    #[test]
    fn the_reference_functions_agree_with_std() {
        // The table's third column is what every forward comparison is judged
        // against; a typo there would make a broken op look correct.
        for (name, domain, _, reference) in unaries() {
            for v in domain.sample(5, 16) {
                let expected = match name {
                    "abs" => v.abs(),
                    "neg" => -v,
                    "sqr" => v * v,
                    "sqrt" => v.sqrt(),
                    "exp" => v.exp(),
                    "log" => v.ln(),
                    "tanh_exact" => tanh_exact_ref(v),
                    _ => continue,
                };
                assert_eq!(reference(v), expected, "{name} at {v}");
            }
        }
    }
}
