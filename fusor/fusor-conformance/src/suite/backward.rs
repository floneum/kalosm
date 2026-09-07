//! The op x backward matrix. Every op with a requires-grad parent must
//! produce a gradient for it; comparisons differentiate to zero rather than to
//! nothing.
//!
//! The distinction that organizes this file: **absent is not zero**. The tape
//! validates that every `Parent { requires_grad: true }` receives a gradient,
//! so a comparison whose adjoint rule is missing and a comparison whose
//! adjoint is the zero tensor are different outcomes, and only the second is
//! correct. [`crate::compare::assert_all_zero`] is the assertion that tells
//! them apart; `gradient_of` erroring out is the other one.

use fusor::tensor::Dyn as Tensor;
use fusor::{Dtype, Session};

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, dims, fuzz_case};
use crate::suite::support::{
    Domain, ELEMENTWISE_SPEC, expect_values, gradient_of, graph_of, loss_of, read, read_probe_loss,
    upload,
};

/// Forward-only rank-2 shapes: no finite differences, so the extents can go
/// past the FD budget.
const FORWARD_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 8), FuzzDim::Range(1, 48)];

/// Analytic-gradient rank-2 shapes: one backward per run, no FD rebuilds.
const ANALYTIC_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(1, 16)];

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

fn len_of(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

fn usize_shape(shape: &[u64]) -> Vec<usize> {
    shape.iter().map(|n| *n as usize).collect()
}

/// The build receives the sampled shape so shape-dependent chains
/// (`chained_views`) stay legal at every run.
type Build = fn(&Tensor, &[u64]) -> fusor::Result<Tensor>;

/// Unary chains whose adjoint is checked against central differences. Each is
/// a composition rather than a bare op, so the case exercises the chain rule
/// through the tape and not just one rule's table row.
#[rustfmt::skip]
fn chains() -> Vec<(&'static str, Build, Domain)> {
    vec![
        ("exp_of_a_product",   |x, _| x.mul(x)?.exp(),                     Domain::Unit),
        ("log_of_a_sum",       |x, _| x.add_scalar(2.0f32)?.log(),         Domain::Unit),
        ("sqrt_of_a_square",   |x, _| x.sqr()?.add_scalar(0.5f32)?.sqrt(), Domain::Wide),
        ("tanh_of_a_scale",    |x, _| x.mul_scalar(2.0f32)?.tanh(),        Domain::Wide),
        ("sigmoid",            |x, _| x.sigmoid(),                         Domain::Wide),
        ("silu",               |x, _| x.silu(),                            Domain::Wide),
        ("gelu",               |x, _| x.gelu(),                            Domain::Wide),
        ("gelu_exact",         |x, _| x.gelu_exact(),                      Domain::Wide),
        ("softplus",           |x, _| x.softplus(),                        Domain::Wide),
        ("leaky_relu",         |x, _| x.leaky_relu(0.1),                   Domain::Wide),
        ("tanh_exact",         |x, _| x.tanh_exact(),                      Domain::Wide),
        ("recip",              |x, _| x.recip(),                           Domain::Positive),
        ("pow_scalar_3",       |x, _| x.pow_scalar(3.0f32),                Domain::Positive),
        ("div_by_a_scalar",    |x, _| x.div_scalar(4.0f32),                Domain::Wide),
        ("rsub_scalar",        |x, _| x.rsub_scalar(1.0f32),               Domain::Wide),
        ("rdiv_scalar",        |x, _| x.rdiv_scalar(2.0f32),               Domain::Positive),
        ("max_scalar",         |x, _| x.max_scalar(0.1f32),                Domain::Wide),
        ("min_scalar",         |x, _| x.min_scalar(0.1f32),                Domain::Wide),
        ("norm_over_an_axis",  |x, _| x.norm(1),                           Domain::Positive),
        ("sum_then_exp",       |x, _| x.sum(1)?.exp(),                     Domain::Unit),
        ("mean_then_square",   |x, _| x.mean(1)?.sqr(),                    Domain::Wide),
        ("chained_views",      |x, s| x.t()?.reshape_dims(&dims(&[s[0] * s[1]]))?.exp(), Domain::Unit),
    ]
}

/// The twelve comparisons, in both their scalar and tensor forms. Each must
/// return a gradient that is present and identically zero.
#[rustfmt::skip]
fn comparisons() -> Vec<(&'static str, Build)> {
    vec![
        ("eq_scalar",  |x, _: &[u64]| x.eq_scalar(0.0f32)),
        ("ne_scalar",  |x, _: &[u64]| x.ne_scalar(0.0f32)),
        ("lt_scalar",  |x, _: &[u64]| x.lt_scalar(0.0f32)),
        ("lte_scalar", |x, _: &[u64]| x.lte_scalar(0.0f32)),
        ("gt_scalar",  |x, _: &[u64]| x.gt_scalar(0.0f32)),
        ("gte_scalar", |x, _: &[u64]| x.gte_scalar(0.0f32)),
        ("eq_tensor",  |x, _: &[u64]| x.eq_tensor(x)),
        ("ne_tensor",  |x, _: &[u64]| x.ne_tensor(x)),
        ("lt_tensor",  |x, _: &[u64]| { let s = x.mul_scalar(2.0f32)?; x.lt_tensor(&s) }),
        ("lte_tensor", |x, _: &[u64]| { let s = x.mul_scalar(2.0f32)?; x.lte_tensor(&s) }),
        ("gt_tensor",  |x, _: &[u64]| { let s = x.mul_scalar(2.0f32)?; x.gt_tensor(&s) }),
        ("gte_tensor", |x, _: &[u64]| { let s = x.mul_scalar(2.0f32)?; x.gte_tensor(&s) }),
    ]
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    for (name, build, domain) in chains() {
        cases.push_case(fuzz_case(
            "backward",
            name,
            ELEMENTWISE_SPEC,
            async move |s: &Session, shape: &[u64], seed: u32| {
                chain_case(s, name, build, domain, shape, seed).await
            },
        ));
    }
    for (name, build) in comparisons() {
        let case: &'static str =
            Box::leak(format!("{name}_differentiates_to_zero").into_boxed_str());
        cases.push_case(fuzz_case(
            "backward",
            case,
            FORWARD_SPEC,
            async move |s: &Session, shape: &[u64], seed: u32| {
                zero_grad_case(s, name, build, shape, seed).await
            },
        ));
    }

    // The clamp data must straddle both bounds, so its width floor keeps at
    // least the three forced elements.
    const CLAMP_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(3, 16)];
    cases.push_case(fuzz_case(
        "backward",
        "clamp_masks_both_ends",
        CLAMP_SPEC,
        clamp_case,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "where_cond_splits_the_gradient",
        FORWARD_SPEC,
        where_cond_case,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "where_cond_gives_the_condition_zeros",
        FORWARD_SPEC,
        where_cond_zero,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "pow_tensor_tensor",
        ANALYTIC_SPEC,
        pow_tensor_case,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "broadcast_add_sums_over_the_stride_zero_axis",
        ANALYTIC_SPEC,
        broadcast_case,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "broadcast_mul_backward",
        ANALYTIC_SPEC,
        broadcast_mul_case,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "gelu_matches_its_analytic_derivative",
        ANALYTIC_SPEC,
        gelu_analytic,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "relu_is_subgradient_zero_at_the_kink",
        ANALYTIC_SPEC,
        relu_kink,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "straight_through_fake_quant",
        ANALYTIC_SPEC,
        straight_through_case,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "detach_cuts_the_tape",
        ANALYTIC_SPEC,
        detach_case,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "an_accumulated_adjoint_fires_once",
        ANALYTIC_SPEC,
        diamond_case,
    ));
    cases.push_case(fuzz_case(
        "backward",
        "backward_seeded_scales_the_whole_gradient",
        ANALYTIC_SPEC,
        seeded_case,
    ));
    cases.push(
        "backward",
        "backward_across_two_graphs_is_refused",
        cross_graph,
    );
    cases.push_case(fuzz_case(
        "backward",
        "a_gradient_reaches_every_requires_grad_parent",
        ANALYTIC_SPEC,
        every_parent,
    ));
    cases
}

/// Forward stays unchecked here — the `elementwise` area owns that — and the
/// adjoint is compared against central differences.
async fn chain_case(
    session: &Session,
    name: &'static str,
    build: Build,
    domain: Domain,
    shape: &[u64],
    seed: u32,
) -> CaseResult {
    let data = domain.sample(seed, len_of(shape));
    let dimv = dims(shape);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = build(&x, shape).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let analytic = gradient_of(&graph, &y, &x).await?;
    let probe_graph = graph_of(session);
    let probe_x = upload(probe_graph.handle(), &dimv, &data)?;
    let probe_y = build(&probe_x, shape).map_err(|e| -> CaseError { e.to_string().into() })?;
    let probe_loss = loss_of(&probe_y)?;
    let numeric = finite_difference_gradient(&usize_shape(shape), &data, |probe| {
        read_probe_loss(&probe_x, &probe_loss, probe)
    })
    .await?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)
        .map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;
    Ok(())
}

/// A comparison's gradient must be **present and zero**. `gradient_of`
/// returning `Err` means no rule fired at all, which the tape treats as an
/// error and so does this case.
async fn zero_grad_case(
    session: &Session,
    name: &'static str,
    build: Build,
    shape: &[u64],
    seed: u32,
) -> CaseResult {
    let data = Domain::Wide.sample(seed, len_of(shape));
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = build(&x, shape).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    // The forward is 1/0 in the operand dtype — no third value.
    let out = read(&y).await?;
    if let Some((i, v)) = out
        .iter()
        .enumerate()
        .find(|(_, v)| **v != 0.0 && **v != 1.0)
    {
        return Err(format!("{name}: element {i} is {v}; a comparison is 1.0 or 0.0").into());
    }

    let grad = gradient_of(&graph, &y, &x)
        .await
        .map_err(|e| -> CaseError {
            format!(
                "{name}: no gradient reached the operand ({e}). A comparison differentiates to \
             zero, not to nothing — every requires-grad parent must receive one."
            )
            .into()
        })?;
    crate::compare::assert_all_zero(name, &grad)?;
    Ok(())
}

/// `clamp`'s adjoint is the `(x > lo) * (x < hi)` mask. The data straddles
/// both bounds, so a rule that masks only one end fails.
async fn clamp_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    const LO: f32 = -0.2;
    const HI: f32 = 0.2;
    let len = len_of(shape);
    let mut data = Domain::Wide.sample(seed, len);
    // Three forced elements so both bounds bite at every sampled shape.
    data[0] = -0.4;
    data[1] = 0.0;
    data[2] = 0.4;
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .clamp(LO, HI)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = data.iter().map(|v| v.clamp(LO, HI)).collect();
    expect_values(session, shape, Dtype::F32, &read(&y).await?, &expected).await?;

    let grad = gradient_of(&graph, &y, &x).await?;
    let want: Vec<f32> = data.iter().map(|v| f32::from(*v > LO && *v < HI)).collect();
    if !want.contains(&0.0) || !want.contains(&1.0) {
        return Err(
            "the clamp case's data does not straddle both bounds; it proves nothing".into(),
        );
    }
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want, &grad, 1e-5, 1e-5)?;
    Ok(())
}

/// `where_cond(cond, a, b)`: `a` receives `grad * mask`, `b` receives
/// `grad * (1 - mask)`.
async fn where_cond_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    let cond_src = Domain::Wide.sample(seed, len);
    let a_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, len);
    let b_data = Domain::Wide.sample(seed.wrapping_add(1), len);

    let graph = graph_of(session);
    let c = upload(graph.handle(), &dims(shape), &cond_src)?;
    let mask = c
        .gt_scalar(0.0f32)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let a = upload(graph.handle(), &dims(shape), &a_data)?;
    let b = upload(graph.handle(), &dims(shape), &b_data)?;
    let y = mask
        .where_cond(&a, &b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let picks: Vec<f32> = cond_src.iter().map(|v| f32::from(*v > 0.0)).collect();
    let expected: Vec<f32> = (0..len)
        .map(|i| {
            if picks[i] == 1.0 {
                a_data[i]
            } else {
                b_data[i]
            }
        })
        .collect();
    expect_values(session, shape, Dtype::F32, &read(&y).await?, &expected).await?;

    let d_a = gradient_of(&graph, &y, &a).await?;
    let d_b = gradient_of(&graph, &y, &b).await?;
    let want_b: Vec<f32> = picks.iter().map(|m| 1.0 - m).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &picks, &d_a, 1e-5, 1e-5)?;
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want_b, &d_b, 1e-5, 1e-5)?;
    Ok(())
}

/// The condition operand receives zeros — present, not absent.
async fn where_cond_zero(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    let cond_src = Domain::Wide.sample(seed, len);
    let graph = graph_of(session);
    let c = upload(graph.handle(), &dims(shape), &cond_src)?;
    let a = upload(
        graph.handle(),
        &dims(shape),
        &Domain::Wide.sample(seed ^ 0x9e37_79b9, len),
    )?;
    let b = upload(
        graph.handle(),
        &dims(shape),
        &Domain::Wide.sample(seed.wrapping_add(1), len),
    )?;
    let y = c
        .where_cond(&a, &b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let d_c = gradient_of(&graph, &y, &c)
        .await
        .map_err(|e| -> CaseError {
            format!("no gradient reached the condition ({e}); it must receive zeros").into()
        })?;
    crate::compare::assert_all_zero("where_cond condition", &d_c)?;
    Ok(())
}

/// `pow(a, b)`: `d_a = b * a^(b-1)`, `d_b = a^b * ln(a)`. The base stays
/// positive so `ln(a)` is defined.
async fn pow_tensor_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    let a_data = Domain::Custom(0.5, 2.0).sample(seed, len);
    let b_data = Domain::Custom(0.5, 2.5).sample(seed ^ 0x9e37_79b9, len);
    let dimv = dims(shape);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dimv, &a_data)?;
    let b = upload(graph.handle(), &dimv, &b_data)?;
    let y = a
        .pow(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = a_data
        .iter()
        .zip(&b_data)
        .map(|(x, y)| x.powf(*y))
        .collect();
    expect_values(session, shape, Dtype::F32, &read(&y).await?, &expected).await?;

    let d_a = gradient_of(&graph, &y, &a).await?;
    let want_a: Vec<f32> = a_data
        .iter()
        .zip(&b_data)
        .map(|(x, e)| e * x.powf(e - 1.0))
        .collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want_a, &d_a, 1e-3, 1e-3)?;

    let d_b = gradient_of(&graph, &y, &b).await?;
    let want_b: Vec<f32> = a_data
        .iter()
        .zip(&b_data)
        .map(|(x, e)| x.powf(*e) * x.ln())
        .collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want_b, &d_b, 1e-3, 1e-3)?;
    Ok(())
}

/// A stride-0 axis's adjoint is a sum over that axis. `[r, c] + [c]` reads
/// each bias element `r` times, so each gets a gradient of `r`.
async fn broadcast_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    let x_data = Domain::Wide.sample(seed, rows * cols);
    let bias = Domain::Wide.sample(seed ^ 0x9e37_79b9, cols);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &x_data)?;
    let b = upload(graph.handle(), &dims(&[cols as u64]), &bias)?;
    let y = x
        .add_(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = x_data
        .iter()
        .enumerate()
        .map(|(i, v)| v + bias[i % cols])
        .collect();
    expect_values(session, shape, Dtype::F32, &read(&y).await?, &expected).await?;

    let d_b = gradient_of(&graph, &y, &b).await?;
    let want = vec![rows as f32; cols];
    crate::compare::approx_or_relative_eq(backend_of(session), &[cols], &want, &d_b, 1e-5, 1e-5)?;
    Ok(())
}

/// The same, through a multiply, where the summed gradient is data dependent
/// rather than a constant.
async fn broadcast_mul_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    let x_data = Domain::Wide.sample(seed, rows * cols);
    let scale = Domain::Custom(0.5, 1.5).sample(seed ^ 0x9e37_79b9, cols);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &x_data)?;
    let s = upload(graph.handle(), &dims(&[cols as u64]), &scale)?;
    let y = x
        .mul_(&s)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let d_s = gradient_of(&graph, &y, &s).await?;
    let want: Vec<f32> = (0..cols)
        .map(|c| (0..rows).map(|r| x_data[r * cols + c]).sum())
        .collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[cols], &want, &d_s, 1e-4, 1e-4)?;

    let d_x = gradient_of(&graph, &y, &x).await?;
    let want_x: Vec<f32> = (0..rows * cols).map(|i| scale[i % cols]).collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[rows * cols],
        &want_x,
        &d_x,
        1e-5,
        1e-5,
    )?;
    Ok(())
}

/// `0.5*(1+t) + 0.5*x*(1-t^2)*c*(1+3*0.044715*x^2)` with
/// `t = tanh(c*(x + 0.044715 x^3))`, `c = sqrt(2/pi)`.
///
/// Checked against the closed form rather than against finite differences:
/// the tanh approximation's derivative is what a rule that differentiates the
/// *exact* gelu would get subtly wrong, and central differences at 1e-3 do
/// not separate the two.
async fn gelu_analytic(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    let data = Domain::Custom(-2.5, 2.5).sample(seed, len);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .gelu()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    // The forward must be the tanh approximation, not the erf one, or the
    // analytic derivative below is being compared against the wrong function.
    let expected: Vec<f32> = data.iter().copied().map(host_gelu).collect();
    expect_values(session, shape, Dtype::F32, &read(&y).await?, &expected).await?;
    let grad = gradient_of(&graph, &y, &x).await?;
    let want: Vec<f32> = data.iter().copied().map(host_gelu_grad).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want, &grad, 2e-3, 2e-3)?;
    Ok(())
}

const GELU_C: f32 = 0.797_884_6; // sqrt(2/pi)
const GELU_A: f32 = 0.044_715;

fn host_gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (GELU_C * (x + GELU_A * x * x * x)).tanh())
}

fn host_gelu_grad(x: f32) -> f32 {
    let t = (GELU_C * (x + GELU_A * x * x * x)).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GELU_C * (1.0 + 3.0 * GELU_A * x * x)
}

/// `relu` is not differentiable at 0. The convention is subgradient 0, and
/// the case pins it: an implementation that answers 1 there makes a dead unit
/// come back to life. The first element is forced to exactly 0 so the kink is
/// present at every sampled shape.
async fn relu_kink(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    let mut data = Domain::Wide.sample(seed, len);
    data[0] = 0.0;
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .relu()
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = data.iter().map(|v| v.max(0.0)).collect();
    expect_values(session, shape, Dtype::F32, &read(&y).await?, &expected).await?;

    let grad = gradient_of(&graph, &y, &x).await?;
    let want: Vec<f32> = data.iter().map(|v| f32::from(*v > 0.0)).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want, &grad, 1e-6, 1e-6)?;
    Ok(())
}

/// QAT: `fake_quant` is opaque forward and the identity backward, and it
/// needs zero user code — the backward it registers carries it.
///
/// Without it the round inside would differentiate to zero everywhere and no
/// quantization-aware model would train at all.
async fn straight_through_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    let data = Domain::Custom(-1.0, 1.0).sample(seed, len);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let scale = upload(graph.handle(), &dims(&[1]), &[0.25f32])?;
    let q = x
        .fake_quant(7, &scale)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = data
        .iter()
        .map(|v| (v / 0.25).round().clamp(-7.0, 7.0) * 0.25)
        .collect();
    expect_values(session, shape, Dtype::F32, &read(&q).await?, &expected).await?;

    let grad = gradient_of(&graph, &q, &x).await?;
    let want = vec![1.0f32; len];
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want, &grad, 1e-6, 1e-6)
        .map_err(|e| -> CaseError {
            format!(
                "{e}: fake_quant must be straight-through. The `round` inside has a zero \
                 derivative, so without it no QAT model trains."
            )
            .into()
        })?;
    Ok(())
}

/// `detach` re-leafs a value, so nothing upstream of it receives a gradient.
async fn detach_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let data = Domain::Wide.sample(seed, len_of(shape));
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let mid = x.sqr().map_err(|e| -> CaseError { e.to_string().into() })?;
    let cut = mid
        .detach()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = cut
        .mul_scalar(3.0f32)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // The detached copy holds the same values...
    let expected: Vec<f32> = data.iter().map(|v| 3.0 * v * v).collect();
    expect_values(session, shape, Dtype::F32, &read(&y).await?, &expected).await?;
    // ...but the tape no longer runs through it.
    if gradient_of(&graph, &y, &x).await.is_ok() {
        return Err("a gradient reached through detach(); it must cut the tape".into());
    }
    Ok(())
}

/// A value consumed twice must have its two adjoints accumulated before its
/// own rule fires — that is what the pending-children counter buys. The
/// gradient of `x*x + x` is `2x + 1`, and a rule that fires on the first
/// adjoint alone gives `x + 1`.
async fn diamond_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    let data = Domain::Wide.sample(seed, len);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .mul(&x)
        .and_then(|sq| sq.add(&x))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &x).await?;
    let want: Vec<f32> = data.iter().map(|v| 2.0 * v + 1.0).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want, &grad, 1e-4, 1e-4)
        .map_err(|e| -> CaseError {
            format!("{e}: a node consumed twice must accumulate both adjoints before it fires")
                .into()
        })?;
    Ok(())
}

/// `backward_seeded` is the loss-scale entry point: a seed of `s` scales every
/// gradient by `s`.
async fn seeded_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    const SCALE: f32 = 8.0;
    let len = len_of(shape);
    let data = Domain::Wide.sample(seed, len);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x.sqr().map_err(|e| -> CaseError { e.to_string().into() })?;
    let loss = loss_of(&y)?;
    let seed_t = upload(graph.handle(), &dims(&[]), &[SCALE]).or_else(|_| {
        // A rank-0 upload may not be expressible; a [1] seed is the same value.
        upload(graph.handle(), &dims(&[1]), &[SCALE])
    })?;
    let grads = graph
        .backward_seeded(&loss, &seed_t, std::slice::from_ref(&x))
        .map_err(|e| -> CaseError { format!("backward_seeded: {e}").into() })?;
    let g = grads
        .get(&x)
        .ok_or_else(|| -> CaseError { "no gradient for the seeded backward".into() })?;
    let got = read(&g).await?;
    let want: Vec<f32> = data.iter().map(|v| SCALE * 2.0 * v).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want, &got, 1e-4, 1e-4)?;
    Ok(())
}

/// Two graphs are two tapes. Differentiating across them is a user error, not
/// a silent zero. An error path, so it stays at a fixed shape.
async fn cross_graph(session: &Session) -> CaseResult {
    const SHAPE: &[u64] = &[3, 4];
    const LEN: usize = 12;
    let a = graph_of(session);
    let b = graph_of(session);
    let x = upload(a.handle(), &dims(SHAPE), &Domain::Wide.sample(1451, LEN))?;
    let other = upload(b.handle(), &dims(SHAPE), &Domain::Wide.sample(1453, LEN))?;
    let loss = loss_of(&x.sqr().map_err(|e| -> CaseError { e.to_string().into() })?)?;
    if b.backward_with(&loss, std::slice::from_ref(&other)).is_ok() {
        return Err("backward accepted a loss from a different graph".into());
    }
    Ok(())
}

/// A multi-input expression must hand a gradient to *every* requires-grad
/// operand. The classic failure is a rule that returns only `d_lhs`. `b` is
/// a divisor, so it stays away from zero.
async fn every_parent(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    let a_data = Domain::Wide.sample(seed, len);
    let b_data = Domain::Positive.sample(seed ^ 0x9e37_79b9, len);
    let c_data = Domain::Wide.sample(seed.wrapping_add(1), len);
    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(shape), &a_data)?;
    let b = upload(graph.handle(), &dims(shape), &b_data)?;
    let c = upload(graph.handle(), &dims(shape), &c_data)?;
    let y = a
        .mul(&b)
        .and_then(|p| p.sub(&c))
        .and_then(|d| d.div(&b))
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    for (label, operand) in [("a", &a), ("b", &b), ("c", &c)] {
        let grad = gradient_of(&graph, &y, operand)
            .await
            .map_err(|e| -> CaseError {
                format!("operand {label} received no gradient: {e}").into()
            })?;
        if grad.len() != len {
            return Err(format!("operand {label}'s gradient has {} elements", grad.len()).into());
        }
        if grad.iter().all(|v| *v == 0.0) {
            return Err(format!("operand {label}'s gradient is identically zero").into());
        }
    }
    Ok(())
}
