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
//!
//! Owned by W14.

use fusor2::{Dtype, Session, Tensor};

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, dims};
use crate::suite::support::{
    Domain, expect_values, gradient_of, graph_of, loss_of, read, read_scalar, upload,
};

const SHAPE: &[u64] = &[3, 4];
const LEN: usize = 12;

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

type Build = fn(&Tensor) -> fusor2::Result<Tensor>;

/// Unary chains whose adjoint is checked against central differences. Each is
/// a composition rather than a bare op, so the case exercises the chain rule
/// through the tape and not just one rule's table row.
#[rustfmt::skip]
fn chains() -> Vec<(&'static str, Build, Domain)> {
    vec![
        ("exp_of_a_product",   |x| x.mul(x)?.exp(),                     Domain::Unit),
        ("log_of_a_sum",       |x| x.add_scalar(2.0f32)?.log(),         Domain::Unit),
        ("sqrt_of_a_square",   |x| x.sqr()?.add_scalar(0.5f32)?.sqrt(), Domain::Wide),
        ("tanh_of_a_scale",    |x| x.mul_scalar(2.0f32)?.tanh(),        Domain::Wide),
        ("sigmoid",            |x| x.sigmoid(),                         Domain::Wide),
        ("silu",               |x| x.silu(),                            Domain::Wide),
        ("gelu",               |x| x.gelu(),                            Domain::Wide),
        ("gelu_exact",         |x| x.gelu_exact(),                      Domain::Wide),
        ("softplus",           |x| x.softplus(),                        Domain::Wide),
        ("leaky_relu",         |x| x.leaky_relu(0.1),                   Domain::Wide),
        ("tanh_exact",         |x| x.tanh_exact(),                      Domain::Wide),
        ("recip",              |x| x.recip(),                           Domain::Positive),
        ("pow_scalar_3",       |x| x.pow_scalar(3.0f32),                Domain::Positive),
        ("div_by_a_scalar",    |x| x.div_scalar(4.0f32),                Domain::Wide),
        ("rsub_scalar",        |x| x.rsub_scalar(1.0f32),               Domain::Wide),
        ("rdiv_scalar",        |x| x.rdiv_scalar(2.0f32),               Domain::Positive),
        ("max_scalar",         |x| x.max_scalar(0.1f32),                Domain::Wide),
        ("min_scalar",         |x| x.min_scalar(0.1f32),                Domain::Wide),
        ("norm_over_an_axis",  |x| x.norm(1),                           Domain::Positive),
        ("sum_then_exp",       |x| x.sum(1)?.exp(),                     Domain::Unit),
        ("mean_then_square",   |x| x.mean(1)?.sqr(),                    Domain::Wide),
        ("chained_views",      |x| x.t()?.reshape_dims(&dims(&[2, 6]))?.exp(), Domain::Unit),
    ]
}

/// The twelve comparisons, in both their scalar and tensor forms. Each must
/// return a gradient that is present and identically zero.
#[rustfmt::skip]
fn comparisons() -> Vec<(&'static str, Build)> {
    vec![
        ("eq_scalar",  |x| x.eq_scalar(0.0f32)),
        ("ne_scalar",  |x| x.ne_scalar(0.0f32)),
        ("lt_scalar",  |x| x.lt_scalar(0.0f32)),
        ("lte_scalar", |x| x.lte_scalar(0.0f32)),
        ("gt_scalar",  |x| x.gt_scalar(0.0f32)),
        ("gte_scalar", |x| x.gte_scalar(0.0f32)),
        ("eq_tensor",  |x| x.eq_tensor(x)),
        ("ne_tensor",  |x| x.ne_tensor(x)),
        ("lt_tensor",  |x| { let s = x.mul_scalar(2.0f32)?; x.lt_tensor(&s) }),
        ("lte_tensor", |x| { let s = x.mul_scalar(2.0f32)?; x.lte_tensor(&s) }),
        ("gt_tensor",  |x| { let s = x.mul_scalar(2.0f32)?; x.gt_tensor(&s) }),
        ("gte_tensor", |x| { let s = x.mul_scalar(2.0f32)?; x.gte_tensor(&s) }),
        ("mt",         |x| x.mt(0.0f32)),
        ("mte",        |x| x.mte(0.0f32)),
    ]
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    for (name, build, domain) in chains() {
        cases.push("backward", name, move |s| {
            chain_case(s, name, build, domain)
        });
    }
    for (name, build) in comparisons() {
        let case = format!("{name}_differentiates_to_zero");
        cases.push("backward", case, move |s| zero_grad_case(s, name, build));
    }

    cases.push("backward", "clamp_masks_both_ends", clamp_case);
    cases.push(
        "backward",
        "where_cond_splits_the_gradient",
        where_cond_case,
    );
    cases.push(
        "backward",
        "where_cond_gives_the_condition_zeros",
        where_cond_zero,
    );
    cases.push("backward", "pow_tensor_tensor", pow_tensor_case);
    cases.push(
        "backward",
        "broadcast_add_sums_over_the_stride_zero_axis",
        broadcast_case,
    );
    cases.push("backward", "broadcast_mul_backward", broadcast_mul_case);
    cases.push(
        "backward",
        "gelu_matches_its_analytic_derivative",
        gelu_analytic,
    );
    cases.push(
        "backward",
        "relu_is_subgradient_zero_at_the_kink",
        relu_kink,
    );
    cases.push(
        "backward",
        "straight_through_fake_quant",
        straight_through_case,
    );
    cases.push("backward", "detach_cuts_the_tape", detach_case);
    cases.push(
        "backward",
        "an_accumulated_adjoint_fires_once",
        diamond_case,
    );
    cases.push(
        "backward",
        "backward_seeded_scales_the_whole_gradient",
        seeded_case,
    );
    cases.push(
        "backward",
        "backward_across_two_graphs_is_refused",
        cross_graph,
    );
    cases.push(
        "backward",
        "a_gradient_reaches_every_requires_grad_parent",
        every_parent,
    );
    cases
}

/// Forward stays unchecked here — the `elementwise` area owns that — and the
/// adjoint is compared against central differences.
fn chain_case(session: &Session, name: &'static str, build: Build, domain: Domain) -> CaseResult {
    let data = domain.sample(1301, LEN);
    let dimv = dims(SHAPE);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = build(&x).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let analytic = gradient_of(&graph, &y, &x)?;
    let numeric = finite_difference_gradient(&[3, 4], &data, &mut |probe| {
        let g = graph_of(session);
        let x = upload(g.handle(), &dimv, probe)?;
        let y = build(&x).map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)
        .map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;
    Ok(())
}

/// A comparison's gradient must be **present and zero**. `gradient_of`
/// returning `Err` means no rule fired at all, which the tape treats as an
/// error and so does this case.
fn zero_grad_case(session: &Session, name: &'static str, build: Build) -> CaseResult {
    let data = Domain::Wide.sample(1303, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let y = build(&x).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    // The forward is 1/0 in the operand dtype — no third value.
    let out = read(&y)?;
    if let Some((i, v)) = out
        .iter()
        .enumerate()
        .find(|(_, v)| **v != 0.0 && **v != 1.0)
    {
        return Err(format!("{name}: element {i} is {v}; a comparison is 1.0 or 0.0").into());
    }

    let grad = gradient_of(&graph, &y, &x).map_err(|e| -> CaseError {
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
fn clamp_case(session: &Session) -> CaseResult {
    const LO: f32 = -0.2;
    const HI: f32 = 0.2;
    let data = Domain::Wide.sample(1307, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let y = x
        .clamp(LO, HI)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = data.iter().map(|v| v.clamp(LO, HI)).collect();
    expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)?;

    let grad = gradient_of(&graph, &y, &x)?;
    let want: Vec<f32> = data.iter().map(|v| f32::from(*v > LO && *v < HI)).collect();
    if !want.iter().any(|v| *v == 0.0) || !want.iter().any(|v| *v == 1.0) {
        return Err(
            "the clamp case's data does not straddle both bounds; it proves nothing".into(),
        );
    }
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want, &grad, 1e-5, 1e-5)?;
    Ok(())
}

/// `where_cond(cond, a, b)`: `a` receives `grad * mask`, `b` receives
/// `grad * (1 - mask)`.
fn where_cond_case(session: &Session) -> CaseResult {
    let cond_src = Domain::Wide.sample(1309, LEN);
    let a_data = Domain::Wide.sample(1319, LEN);
    let b_data = Domain::Wide.sample(1321, LEN);

    let graph = graph_of(session);
    let c = upload(graph.handle(), &dims(SHAPE), &cond_src)?;
    let mask = c
        .gt_scalar(0.0f32)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let a = upload(graph.handle(), &dims(SHAPE), &a_data)?;
    let b = upload(graph.handle(), &dims(SHAPE), &b_data)?;
    let y = mask
        .where_cond(&a, &b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let picks: Vec<f32> = cond_src.iter().map(|v| f32::from(*v > 0.0)).collect();
    let expected: Vec<f32> = (0..LEN)
        .map(|i| {
            if picks[i] == 1.0 {
                a_data[i]
            } else {
                b_data[i]
            }
        })
        .collect();
    expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)?;

    let d_a = gradient_of(&graph, &y, &a)?;
    let d_b = gradient_of(&graph, &y, &b)?;
    let want_b: Vec<f32> = picks.iter().map(|m| 1.0 - m).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &picks, &d_a, 1e-5, 1e-5)?;
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want_b, &d_b, 1e-5, 1e-5)?;
    Ok(())
}

/// The condition operand receives zeros — present, not absent.
fn where_cond_zero(session: &Session) -> CaseResult {
    let cond_src = Domain::Wide.sample(1327, LEN);
    let graph = graph_of(session);
    let c = upload(graph.handle(), &dims(SHAPE), &cond_src)?;
    let a = upload(
        graph.handle(),
        &dims(SHAPE),
        &Domain::Wide.sample(1331, LEN),
    )?;
    let b = upload(
        graph.handle(),
        &dims(SHAPE),
        &Domain::Wide.sample(1361, LEN),
    )?;
    let y = c
        .where_cond(&a, &b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let d_c = gradient_of(&graph, &y, &c).map_err(|e| -> CaseError {
        format!("no gradient reached the condition ({e}); it must receive zeros").into()
    })?;
    crate::compare::assert_all_zero("where_cond condition", &d_c)?;
    Ok(())
}

/// `pow(a, b)`: `d_a = b * a^(b-1)`, `d_b = a^b * ln(a)`. The base stays
/// positive so `ln(a)` is defined.
fn pow_tensor_case(session: &Session) -> CaseResult {
    let a_data = Domain::Custom(0.5, 2.0).sample(1367, LEN);
    let b_data = Domain::Custom(0.5, 2.5).sample(1373, LEN);
    let dimv = dims(SHAPE);

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
    expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)?;

    let d_a = gradient_of(&graph, &y, &a)?;
    let want_a: Vec<f32> = a_data
        .iter()
        .zip(&b_data)
        .map(|(x, e)| e * x.powf(e - 1.0))
        .collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want_a, &d_a, 1e-3, 1e-3)?;

    let d_b = gradient_of(&graph, &y, &b)?;
    let want_b: Vec<f32> = a_data
        .iter()
        .zip(&b_data)
        .map(|(x, e)| x.powf(*e) * x.ln())
        .collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want_b, &d_b, 1e-3, 1e-3)?;
    Ok(())
}

/// A stride-0 axis's adjoint is a sum over that axis. `[3, 4] + [4]` reads
/// each bias element three times, so each gets a gradient of 3.
fn broadcast_case(session: &Session) -> CaseResult {
    let x_data = Domain::Wide.sample(1381, LEN);
    let bias = Domain::Wide.sample(1399, 4);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &x_data)?;
    let b = upload(graph.handle(), &dims(&[4]), &bias)?;
    let y = x
        .broadcast_add(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = x_data
        .iter()
        .enumerate()
        .map(|(i, v)| v + bias[i % 4])
        .collect();
    expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)?;

    let d_b = gradient_of(&graph, &y, &b)?;
    let want = vec![3.0f32; 4];
    crate::compare::approx_or_relative_eq(backend_of(session), &[4], &want, &d_b, 1e-5, 1e-5)?;
    Ok(())
}

/// The same, through a multiply, where the summed gradient is data dependent
/// rather than a constant.
fn broadcast_mul_case(session: &Session) -> CaseResult {
    let x_data = Domain::Wide.sample(1409, LEN);
    let scale = Domain::Custom(0.5, 1.5).sample(1423, 4);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &x_data)?;
    let s = upload(graph.handle(), &dims(&[4]), &scale)?;
    let y = x
        .broadcast_mul(&s)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let d_s = gradient_of(&graph, &y, &s)?;
    let want: Vec<f32> = (0..4)
        .map(|c| (0..3).map(|r| x_data[r * 4 + c]).sum())
        .collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[4], &want, &d_s, 1e-4, 1e-4)?;

    let d_x = gradient_of(&graph, &y, &x)?;
    let want_x: Vec<f32> = (0..LEN).map(|i| scale[i % 4]).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want_x, &d_x, 1e-5, 1e-5)?;
    Ok(())
}

/// `0.5*(1+t) + 0.5*x*(1-t^2)*c*(1+3*0.044715*x^2)` with
/// `t = tanh(c*(x + 0.044715 x^3))`, `c = sqrt(2/pi)`.
///
/// Checked against the closed form rather than against finite differences:
/// the tanh approximation's derivative is what a rule that differentiates the
/// *exact* gelu would get subtly wrong, and central differences at 1e-3 do
/// not separate the two.
fn gelu_analytic(session: &Session) -> CaseResult {
    let data = Domain::Custom(-2.5, 2.5).sample(1427, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let y = x
        .gelu()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    // The forward must be the tanh approximation, not the erf one, or the
    // analytic derivative below is being compared against the wrong function.
    let expected: Vec<f32> = data.iter().copied().map(host_gelu).collect();
    expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)?;
    let grad = gradient_of(&graph, &y, &x)?;
    let want: Vec<f32> = data.iter().copied().map(host_gelu_grad).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want, &grad, 2e-3, 2e-3)?;
    Ok(())
}

const GELU_C: f32 = 0.797_884_56; // sqrt(2/pi)
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
/// come back to life.
fn relu_kink(session: &Session) -> CaseResult {
    let data: Vec<f32> = vec![
        -1.0, -0.5, 0.0, 0.5, 1.0, -2.0, 2.0, 0.0, 0.25, -0.25, 3.0, -3.0,
    ];
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let y = x
        .relu()
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = data.iter().map(|v| v.max(0.0)).collect();
    expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)?;

    let grad = gradient_of(&graph, &y, &x)?;
    let want: Vec<f32> = data.iter().map(|v| f32::from(*v > 0.0)).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want, &grad, 1e-6, 1e-6)?;
    Ok(())
}

/// QAT: `fake_quant` is opaque forward and the identity backward, and it
/// needs zero user code — the backward it registers carries it.
///
/// Without it the round inside would differentiate to zero everywhere and no
/// quantization-aware model would train at all.
fn straight_through_case(session: &Session) -> CaseResult {
    let data = Domain::Custom(-1.0, 1.0).sample(1429, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let scale = upload(graph.handle(), &dims(&[1]), &[0.25f32])?;
    let q = x
        .fake_quant(7, &scale)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = data
        .iter()
        .map(|v| (v / 0.25).round().clamp(-7.0, 7.0) * 0.25)
        .collect();
    expect_values(session, SHAPE, Dtype::F32, &read(&q)?, &expected)?;

    let grad = gradient_of(&graph, &q, &x)?;
    let want = vec![1.0f32; LEN];
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want, &grad, 1e-6, 1e-6)
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
fn detach_case(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(1433, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let mid = x.sqr().map_err(|e| -> CaseError { e.to_string().into() })?;
    let cut = mid
        .detach()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = cut
        .mul_scalar(3.0f32)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // The detached copy holds the same values...
    let expected: Vec<f32> = data.iter().map(|v| 3.0 * v * v).collect();
    expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)?;
    // ...but the tape no longer runs through it.
    if gradient_of(&graph, &y, &x).is_ok() {
        return Err("a gradient reached through detach(); it must cut the tape".into());
    }
    Ok(())
}

/// A value consumed twice must have its two adjoints accumulated before its
/// own rule fires — that is what the pending-children counter buys. The
/// gradient of `x*x + x` is `2x + 1`, and a rule that fires on the first
/// adjoint alone gives `x + 1`.
fn diamond_case(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(1439, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let y = x
        .mul(&x)
        .and_then(|sq| sq.add(&x))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &x)?;
    let want: Vec<f32> = data.iter().map(|v| 2.0 * v + 1.0).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want, &grad, 1e-4, 1e-4)
        .map_err(|e| -> CaseError {
            format!("{e}: a node consumed twice must accumulate both adjoints before it fires")
                .into()
        })?;
    Ok(())
}

/// `backward_seeded` is the loss-scale entry point: a seed of `s` scales every
/// gradient by `s`.
fn seeded_case(session: &Session) -> CaseResult {
    const SCALE: f32 = 8.0;
    let data = Domain::Wide.sample(1447, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let y = x.sqr().map_err(|e| -> CaseError { e.to_string().into() })?;
    let loss = loss_of(&y)?;
    let seed = upload(graph.handle(), &dims(&[]), &[SCALE]).or_else(|_| {
        // A rank-0 upload may not be expressible; a [1] seed is the same value.
        upload(graph.handle(), &dims(&[1]), &[SCALE])
    })?;
    let grads = graph
        .backward_seeded(&loss, &seed, std::slice::from_ref(&x))
        .map_err(|e| -> CaseError { format!("backward_seeded: {e}").into() })?;
    let g = grads
        .get(&x)
        .ok_or_else(|| -> CaseError { "no gradient for the seeded backward".into() })?;
    let got = read(&g)?;
    let want: Vec<f32> = data.iter().map(|v| SCALE * 2.0 * v).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want, &got, 1e-4, 1e-4)?;
    Ok(())
}

/// Two graphs are two tapes. Differentiating across them is a user error, not
/// a silent zero.
fn cross_graph(session: &Session) -> CaseResult {
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
/// operand. The classic failure is a rule that returns only `d_lhs`.
fn every_parent(session: &Session) -> CaseResult {
    let a_data = Domain::Wide.sample(1459, LEN);
    let b_data = Domain::Wide.sample(1471, LEN);
    let c_data = Domain::Wide.sample(1481, LEN);
    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(SHAPE), &a_data)?;
    let b = upload(graph.handle(), &dims(SHAPE), &b_data)?;
    let c = upload(graph.handle(), &dims(SHAPE), &c_data)?;
    let y = a
        .mul(&b)
        .and_then(|p| p.sub(&c))
        .and_then(|d| d.div(&b))
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    for (label, operand) in [("a", &a), ("b", &b), ("c", &c)] {
        let grad = gradient_of(&graph, &y, operand).map_err(|e| -> CaseError {
            format!("operand {label} received no gradient: {e}").into()
        })?;
        if grad.len() != LEN {
            return Err(format!("operand {label}'s gradient has {} elements", grad.len()).into());
        }
        if grad.iter().all(|v| *v == 0.0) {
            return Err(format!("operand {label}'s gradient is identically zero").into());
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

    #[test]
    fn all_twelve_comparisons_differentiate_to_zero_somewhere_in_the_table() {
        let names = registered();
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
        ] {
            let case = format!("backward::{wanted}_differentiates_to_zero");
            assert!(names.iter().any(|n| *n == case), "{case} is missing");
        }
        assert_eq!(comparisons().len(), 14, "the twelve plus mt/mte");
    }

    #[test]
    fn the_structural_backward_cases_are_registered() {
        let names = registered();
        for wanted in [
            "clamp_masks_both_ends",
            "where_cond_splits_the_gradient",
            "where_cond_gives_the_condition_zeros",
            "straight_through_fake_quant",
            "detach_cuts_the_tape",
            "an_accumulated_adjoint_fires_once",
            "a_gradient_reaches_every_requires_grad_parent",
        ] {
            assert!(
                names.iter().any(|n| n == &format!("backward::{wanted}")),
                "{wanted} is missing"
            );
        }
    }

    #[test]
    fn every_chain_name_is_distinct() {
        let mut names: Vec<&str> = chains().into_iter().map(|(n, _, _)| n).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(before, names.len());
        assert!(before >= 20, "the chain table has only {before} rows");
    }

    #[test]
    fn the_gelu_derivative_is_the_derivative_of_the_gelu() {
        // The analytic form must agree with a finite difference of the same
        // approximation, or the case would be pinning a typo.
        let eps = 1e-3f32;
        for x in [-2.0f32, -0.5, 0.0, 0.5, 1.0, 2.5] {
            let numeric = (host_gelu(x + eps) - host_gelu(x - eps)) / (2.0 * eps);
            let analytic = host_gelu_grad(x);
            assert!(
                (numeric - analytic).abs() < 2e-3,
                "x={x}: analytic {analytic} vs numeric {numeric}"
            );
        }
    }

    #[test]
    fn the_gelu_constants_are_the_published_ones() {
        assert!((GELU_C - (2.0f32 / std::f32::consts::PI).sqrt()).abs() < 1e-6);
        assert_eq!(GELU_A, 0.044_715);
        // gelu(0) = 0 and gelu is monotone over the sampled range.
        assert_eq!(host_gelu(0.0), 0.0);
        assert!(host_gelu(1.0) > host_gelu(0.5));
        assert!(host_gelu_grad(0.0) > 0.0);
    }

    #[test]
    fn the_clamp_bounds_actually_bite_on_the_sampled_data() {
        // Guards the in-case check: with Domain::Wide over [-0.5, 0.5) and
        // bounds at +-0.2 both branches must be populated.
        let data = Domain::Wide.sample(1307, LEN);
        assert!(data.iter().any(|v| *v > 0.2 || *v < -0.2));
        assert!(data.iter().any(|v| *v > -0.2 && *v < 0.2));
    }
}
