//! The 23 unaries, 8 scalar-arith ops, the same-rank and broadcasting
//! binaries, the std-ops surface, `where_cond` and all 12 comparisons.
//!
//! Every one of these is one `Map` with a different `ScalarExpr`, so this
//! suite is really a `ScalarExpr` test.

use fusor::tensor::Dyn as Tensor;
use fusor::{Dtype, Session};

use crate::compare::{
    assert_all_zero, assert_gradient_matches_finite_difference, finite_difference_gradient,
    relative_eq,
};
use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, dense_len, dims, fuzz_case, is_gpu};
use crate::suite::support::{
    BinaryOp, Domain, ELEMENTWISE_SPEC, UnaryOp, binary_case, comparison_case, expect_values,
    gradient_of, graph_of, loss_of, read, read_probe_loss, unary_case, upload,
};

/// The forward-only rows take no gradient, so they can afford multi-workgroup
/// extents. [`non_vacuous`] needs enough samples that a random draw cannot
/// land entirely on the op's identity interval.
/// `(name, sampling domain, op, host reference)` for a unary row that is
/// checked forward and backward.
type UnaryRow = (&'static str, Domain, UnaryOp, fn(f32) -> f32);
/// The binary counterpart of [`UnaryRow`].
type BinaryRow = (&'static str, Domain, BinaryOp, fn(f32, f32) -> f32);
/// `(name, op, host reference)` for a unary row over the default domain.
type UnaryRefRow = (&'static str, UnaryOp, fn(f32) -> f32);
/// The binary counterpart of [`UnaryRefRow`].
type BinaryRefRow = (&'static str, BinaryOp, fn(f32, f32) -> f32);

const FORWARD_SPEC: &[FuzzDim] = &[FuzzDim::Range(4, 8), FuzzDim::Range(8, 64)];

/// `(e^x - e^-x) / (e^x + e^-x)`, the form `tanh_exact` names. The reference
/// needs it where a driver's native tanh under-saturates the GELU tail.
fn tanh_exact_ref(x: f32) -> f32 {
    let (up, down) = (x.exp(), (-x).exp());
    (up - down) / (up + down)
}

/// The 21 unaries with an exact elementwise reference. `approximate_exp` and
/// `less_approximate_exp` are the other two of the 23; they get a relative
/// bound instead.
#[rustfmt::skip]
fn unaries() -> Vec<UnaryRow> {
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
        ("sqr",        Domain::Wide,             |x| x.sqr(),    |v| v * v),
        ("sqrt",       Domain::Positive,         |x| x.sqrt(),   f32::sqrt),
    ]
}

/// The 8 scalar-arith unaries, each with its constant baked into the closure.
#[rustfmt::skip]
fn scalar_arith() -> Vec<UnaryRow> {
    vec![
        ("add_scalar", Domain::Wide,     |x| x.add_scalar(0.75), |v| v + 0.75),
        ("sub_scalar", Domain::Wide,     |x| x.sub_scalar(0.25), |v| v - 0.25),
        ("mul_scalar", Domain::Wide,     |x| x.mul_scalar(-1.5), |v| v * -1.5),
        ("div_scalar", Domain::Wide,     |x| x.div_scalar(2.0),  |v| v / 2.0),
        ("pow_scalar", Domain::Positive, |x| x.pow_scalar(1.5),  |v| v.powf(1.5)),
        // Sampled off the kink so finite differences agree with the adjoint.
        // On [0.2, 1.5) all three are the identity, so these rows check the
        // adjoint only; `forward_only()` owns the clamping half.
        ("max_scalar", Domain::Custom(0.2, 1.5), |x| x.max_scalar(0.1), |v| v.max(0.1)),
        ("min_scalar", Domain::Custom(0.2, 1.5), |x| x.min_scalar(1.9), |v| v.min(1.9)),
        ("clamp",      Domain::Custom(0.2, 1.5), |x| x.clamp(0.1, 1.9), |v| v.clamp(0.1, 1.9)),
    ]
}

/// Forward-only rows, over domains that straddle the kink the tables above
/// sample away from: `abs` on a negative, and each clamp actually clamping.
/// No gradient is taken, so the kink is harmless; [`non_vacuous`] keeps the
/// domain honest.
#[rustfmt::skip]
fn forward_only() -> Vec<UnaryRow> {
    vec![
        ("abs_straddles_zero",       Domain::Wide,            |x| x.abs(),           f32::abs),
        ("max_scalar_clamps_below",  Domain::Custom(-2.0, 2.0), |x| x.max_scalar(0.5), |v| v.max(0.5)),
        ("min_scalar_clamps_above",  Domain::Custom(-2.0, 2.0), |x| x.min_scalar(0.5), |v| v.min(0.5)),
        ("clamp_clamps_both_ends",   Domain::Custom(-2.0, 2.0), |x| x.clamp(-0.5, 0.5), |v| v.clamp(-0.5, 0.5)),
    ]
}

/// Refuse a row whose reference is the identity on its own sampled data:
/// such a case passes against a passthrough implementation.
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
fn binaries() -> Vec<BinaryRow> {
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
fn scalar_comparisons() -> Vec<UnaryRefRow> {
    vec![
        ("eq_scalar",  |x| x.eq_scalar(0.0), |v| f32::from(v == 0.0)),
        ("ne_scalar",  |x| x.ne_scalar(0.0), |v| f32::from(v != 0.0)),
        ("lt_scalar",  |x| x.lt_scalar(0.0), |v| f32::from(v < 0.0)),
        ("lte_scalar", |x| x.lte_scalar(0.0), |v| f32::from(v <= 0.0)),
        ("gt_scalar",  |x| x.gt_scalar(0.0), |v| f32::from(v > 0.0)),
        ("gte_scalar", |x| x.gte_scalar(0.0), |v| f32::from(v >= 0.0)),
    ]
}

/// The 6 tensor comparisons.
#[rustfmt::skip]
fn tensor_comparisons() -> Vec<BinaryRefRow> {
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
fn broadcasting() -> Vec<BinaryRefRow> {
    vec![
        ("add_", |a, b| a.add_(b), |x, y| x + y),
        ("sub_", |a, b| a.sub_(b), |x, y| x - y),
        ("mul_", |a, b| a.mul_(b), |x, y| x * y),
        ("div_", |a, b| a.div_(b), |x, y| x / y),
        ("pow_", |a, b| a.pow_(b), |x, y| x.powf(y)),
    ]
}

/// The forward bound a GPU gets on a row whose driver implementation is a
/// documented approximation, or `None` for the F32 default.
///
/// Mesa lowers the SPIR-V `asin`/`acos` to a fixed polynomial a few 1e-4
/// off in absolute terms — on lavapipe in CI and on every Mesa Vulkan driver
/// in the field alike. The kernel emits the plain `asin`/`acos` builtin, so
/// what is being measured there is the driver, not the compiler; the row's
/// adjoint check (`1 / sqrt(1 - x^2)`, computed from primitives) keeps the
/// default bound.
fn gpu_forward_tolerance(name: &str) -> Option<(f32, f32)> {
    match name {
        "asin" | "acos" => Some((1e-3, 1e-4)),
        _ => None,
    }
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    for (name, domain, op, reference) in unaries() {
        cases.push_case(unary_case(
            "elementwise",
            name,
            ELEMENTWISE_SPEC,
            domain,
            op,
            reference,
            gpu_forward_tolerance(name),
        ));
    }
    for (name, domain, op, reference) in scalar_arith() {
        cases.push_case(unary_case(
            "elementwise",
            name,
            ELEMENTWISE_SPEC,
            domain,
            op,
            reference,
            None,
        ));
    }
    for (name, domain, op, reference) in forward_only() {
        cases.push_case(fuzz_case(
            "elementwise",
            name,
            FORWARD_SPEC,
            move |session, shape, seed| {
                let data = domain.sample(seed, dense_len(&dims(shape)));
                non_vacuous(name, &data, reference)?;
                let graph = graph_of(session);
                let x = upload(graph.handle(), &dims(shape), &data)?;
                let y = op(&x).map_err(|e| -> CaseError { e.to_string().into() })?;
                let expected: Vec<f32> = data.iter().copied().map(reference).collect();
                expect_values(session, shape, Dtype::F32, &read(&y)?, &expected)
            },
        ));
    }
    for (name, domain, op, reference) in binaries() {
        cases.push_case(binary_case(
            "elementwise",
            name,
            ELEMENTWISE_SPEC,
            domain,
            op,
            reference,
        ));
    }
    for (name, op, reference) in scalar_comparisons() {
        cases.push_case(comparison_case("elementwise", name, op, reference));
    }
    for (name, op, reference) in tensor_comparisons() {
        cases.push_case(fuzz_case(
            "elementwise",
            name,
            ELEMENTWISE_SPEC,
            move |session, shape, seed| {
                tensor_comparison_case(session, name, shape, seed, op, reference)
            },
        ));
    }
    for (name, op, reference) in broadcasting() {
        cases.push_case(fuzz_case(
            "elementwise",
            name,
            ELEMENTWISE_SPEC,
            move |session, shape, seed| broadcast_case(session, shape, seed, op, reference),
        ));
    }

    // The two GPU-approximate exponentials get a relative bound rather than
    // an elementwise reference.
    cases.push_case(fuzz_case(
        "elementwise",
        "approximate_exp",
        ELEMENTWISE_SPEC,
        |session, shape, seed| approximate_exp_case(session, "approximate_exp", shape, seed, 5e-3),
    ));
    cases.push_case(fuzz_case(
        "elementwise",
        "less_approximate_exp",
        ELEMENTWISE_SPEC,
        |session, shape, seed| {
            approximate_exp_case(session, "less_approximate_exp", shape, seed, 5e-2)
        },
    ));

    // The two elementwise extrema, whose adjoint is a mask rather than zero.
    cases.push_case(binary_case(
        "elementwise",
        "max_elementwise",
        ELEMENTWISE_SPEC,
        Domain::Wide,
        |a, b| a.maximum(b),
        f32::max,
    ));
    cases.push_case(binary_case(
        "elementwise",
        "min_elementwise",
        ELEMENTWISE_SPEC,
        Domain::Wide,
        |a, b| a.minimum(b),
        f32::min,
    ));

    // A chained expression is a different `ScalarExpr::compose` shape than a
    // single op.
    cases.push_case(fuzz_case(
        "elementwise",
        "std_ops_add_sub",
        ELEMENTWISE_SPEC,
        |s, shape, seed| expr_case(s, shape, seed, |a, b| a.add(b)?.sub(b), |x, y| (x + y) - y),
    ));
    cases.push_case(fuzz_case(
        "elementwise",
        "std_ops_mul_div",
        ELEMENTWISE_SPEC,
        |s, shape, seed| expr_case(s, shape, seed, |a, b| a.mul(b)?.div(b), |x, y| (x * y) / y),
    ));
    cases.push_case(fuzz_case(
        "elementwise",
        "std_ops_neg",
        ELEMENTWISE_SPEC,
        |s, shape, seed| expr_case(s, shape, seed, |a, b| a.neg()?.sub(b), |x, y| -x - y),
    ));
    cases.push_case(fuzz_case(
        "elementwise",
        "std_ops_scalar",
        ELEMENTWISE_SPEC,
        |s, shape, seed| {
            expr_case(
                s,
                shape,
                seed,
                |a, b| a.mul_scalar(3.0)?.add_scalar(-1.0)?.sub(b),
                |x, y| (x * 3.0 - 1.0) - y,
            )
        },
    ));

    cases.push_case(fuzz_case(
        "elementwise",
        "where_cond",
        ELEMENTWISE_SPEC,
        where_cond_case,
    ));
    cases
}

fn backend_of(session: &Session) -> &'static str {
    if is_gpu(session) { "gpu" } else { "cpu" }
}

/// A tensor-tensor comparison: 1.0/0.0 forward, zero gradient to **both**
/// parents. The tape validates that every requires-grad parent receives a
/// gradient, so an absent rule and a zero rule are different outcomes.
fn tensor_comparison_case(
    session: &Session,
    name: &'static str,
    shape: &[u64],
    seed: u32,
    op: BinaryOp,
    reference: fn(f32, f32) -> f32,
) -> CaseResult {
    let len = dense_len(&dims(shape));
    let lhs = Domain::Wide.sample(seed, len);
    // Half the rows share a value with `lhs`, so the equality comparisons are
    // not vacuously all-zero.
    let mut rhs = Domain::Wide.sample(seed ^ 0x9e37_79b9, len);
    for i in (0..len).step_by(2) {
        rhs[i] = lhs[i];
    }
    let dimv = dims(shape);

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
    expect_values(session, shape, Dtype::F32, &actual, &expected)?;

    assert_all_zero(name, &gradient_of(&graph, &y, &a)?)?;
    assert_all_zero(name, &gradient_of(&graph, &y, &b)?)?;
    Ok(())
}

/// A rank-2 activation against a rank-1 operand, right-aligned.
///
/// No implicit broadcasting exists at Logical — the frontend emits
/// `Restride { multiplier: 0 }` — so this tests that the frontend's
/// right-aligned rules hold and that a stride-0 axis's adjoint is a sum over
/// that axis.
fn broadcast_case(
    session: &Session,
    shape: &[u64],
    seed: u32,
    op: BinaryOp,
    reference: fn(f32, f32) -> f32,
) -> CaseResult {
    let (rows, cols) = (shape[0], shape[1]);
    let lhs = Domain::Positive.sample(seed, (rows * cols) as usize);
    let rhs = Domain::Positive.sample(seed ^ 0x9e37_79b9, cols as usize);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[rows, cols]), &lhs)?;
    let b = upload(graph.handle(), &dims(&[cols]), &rhs)?;
    let y = op(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let expected: Vec<f32> = (0..(rows * cols) as usize)
        .map(|i| reference(lhs[i], rhs[i % cols as usize]))
        .collect();
    expect_values(session, &[rows, cols], Dtype::F32, &actual, &expected)?;

    let d_rhs = gradient_of(&graph, &y, &b)?;
    if d_rhs.len() != cols as usize {
        return Err(format!(
            "the broadcast operand's gradient has {} elements, not {cols}: a stride-0 \
             axis's adjoint is a sum over that axis",
            d_rhs.len()
        )
        .into());
    }
    let probe_graph = graph_of(session);
    let probe_a = upload(probe_graph.handle(), &dims(&[rows, cols]), &lhs)?;
    let probe_b = upload(probe_graph.handle(), &dims(&[cols]), &rhs)?;
    let probe_y = op(&probe_a, &probe_b).map_err(|e| -> CaseError { e.to_string().into() })?;
    let probe_loss = loss_of(&probe_y)?;
    let numeric = finite_difference_gradient(&[cols as usize], &rhs, &mut |probe| {
        read_probe_loss(&probe_b, &probe_loss, probe)
    })?;
    assert_gradient_matches_finite_difference(&d_rhs, &numeric)?;
    Ok(())
}

/// A two-operand expression checked forward and on the left gradient.
fn expr_case(
    session: &Session,
    shape: &[u64],
    seed: u32,
    build: fn(&Tensor, &Tensor) -> fusor::Result<Tensor>,
    reference: fn(f32, f32) -> f32,
) -> CaseResult {
    let len = dense_len(&dims(shape));
    let lhs = Domain::Positive.sample(seed, len);
    let rhs = Domain::Positive.sample(seed ^ 0x9e37_79b9, len);
    let dimv = dims(shape);

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
    expect_values(session, shape, Dtype::F32, &actual, &expected)?;

    let analytic = gradient_of(&graph, &y, &a)?;
    let probe_graph = graph_of(session);
    let probe_a = upload(probe_graph.handle(), &dimv, &lhs)?;
    let probe_b = upload(probe_graph.handle(), &dimv, &rhs)?;
    let probe_y = build(&probe_a, &probe_b).map_err(|e| -> CaseError { e.to_string().into() })?;
    let probe_loss = loss_of(&probe_y)?;
    let numeric = finite_difference_gradient(&[len], &lhs, &mut |probe| {
        read_probe_loss(&probe_a, &probe_loss, probe)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

/// An approximate exponential: within `tol` of `exp` in relative terms, and
/// differentiable to itself.
fn approximate_exp_case(
    session: &Session,
    name: &'static str,
    shape: &[u64],
    seed: u32,
    tol: f32,
) -> CaseResult {
    let len = dense_len(&dims(shape));
    let data = Domain::Wide.sample(seed, len);
    let dimv = dims(shape);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;

    let Some(y) = approximate_exp_op(&x, name) else {
        return Err(format!(
            "fusor::Tensor has no `{name}`; it is one of the 23 unaries and must be \
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

/// Resolve the approximate-exponential entry point by name. Both are their
/// own `UnOp`, not aliases for `exp`.
fn approximate_exp_op(x: &Tensor, name: &str) -> Option<fusor::Result<Tensor>> {
    match name {
        "approximate_exp" => Some(x.approximate_exp()),
        "less_approximate_exp" => Some(x.less_approximate_exp()),
        _ => None,
    }
}

/// `where_cond`: condition, on_true and on_false all share one shape and one
/// dtype, because there is no bool. The condition receives a zero gradient and
/// the branches receive the mask and its complement.
fn where_cond_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = dense_len(&dims(shape));
    let cond: Vec<f32> = Domain::Wide
        .sample(seed, len)
        .iter()
        .map(|v| f32::from(*v > 0.0))
        .collect();
    let on_true = Domain::Wide.sample(seed ^ 0x9e37_79b9, len);
    let on_false = Domain::Wide.sample(seed.wrapping_add(1), len);
    let dimv = dims(shape);

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
    expect_values(session, shape, Dtype::F32, &actual, &expected)?;

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
