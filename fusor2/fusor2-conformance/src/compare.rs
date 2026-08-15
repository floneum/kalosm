//! Numeric comparison, per-dtype tolerance, finite differences and backend
//! parity.
//!
//! Ported from `conformance/src/comparison.rs`, with the const-generic rank
//! erased: `fusor2::Tensor` has runtime rank, so a comparison takes a flat
//! slice plus the shape and reports a multi-dimensional index on failure.

use std::fmt::{self, Debug, Display};

use fusor2::Dtype;

use crate::harness::CaseError;

// ---------------------------------------------------------------------------
// Tolerance
// ---------------------------------------------------------------------------

/// `(dtype, absolute, relative)`. Integer dtypes compare exactly — a `U32`
/// index or an `I32` sort key that is off by one is a bug, never roundoff.
pub const DTYPE_TOL: &[(Dtype, f32, f32)] = &[
    (Dtype::F32, 1e-4, 1e-5),
    (Dtype::F16, 2e-2, 5e-3),
    (Dtype::BF16, 6e-2, 2e-2),
    (Dtype::U32, 0.0, 0.0),
    (Dtype::I32, 0.0, 0.0),
];

/// The `(absolute, relative)` tolerance for `dtype`. Quantized values are
/// compared at their dequantized dtype, so `Q(..)` falls back to F32.
pub fn tol_for(dtype: Dtype) -> (f32, f32) {
    DTYPE_TOL
        .iter()
        .find(|(d, _, _)| *d == dtype)
        .map(|(_, a, r)| (*a, *r))
        .unwrap_or((1e-4, 1e-5))
}

/// True when `dtype` must match exactly.
pub fn is_exact(dtype: Dtype) -> bool {
    let (a, r) = tol_for(dtype);
    a == 0.0 && r == 0.0
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// One element disagreed. Carries the backend so a CPU/GPU parity failure
/// names which side produced the bad value.
#[derive(Debug, Clone)]
pub struct ItemMismatchError {
    pub device: String,
    pub position: Vec<usize>,
    pub expected: String,
    pub actual: String,
}

impl ItemMismatchError {
    pub fn new(
        device: impl Into<String>,
        position: impl IntoIterator<Item = usize>,
        expected: impl ToString,
        actual: impl ToString,
    ) -> Self {
        Self {
            device: device.into(),
            position: position.into_iter().collect(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }
}

impl Display for ItemMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let position = if self.position.is_empty() {
            String::from("<scalar>")
        } else {
            format!("{:?}", self.position)
        };
        write!(
            f,
            "item mismatch on {} at {}: expected {}, got {}",
            self.device, position, self.expected, self.actual
        )
    }
}

impl std::error::Error for ItemMismatchError {}

// ---------------------------------------------------------------------------
// Comparators
// ---------------------------------------------------------------------------

/// Row-major index of `flat` in `shape`. An empty shape gives an empty index,
/// which [`ItemMismatchError`] renders as `<scalar>`.
fn index_of(shape: &[usize], flat: usize) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    let mut rem = flat;
    for d in (0..shape.len()).rev() {
        let extent = shape[d].max(1);
        idx[d] = rem % extent;
        rem /= extent;
    }
    idx
}

/// The general form: compare elementwise under `eq`, reporting the first
/// offender's multi-dimensional index.
pub fn eq_with(
    device: &str,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
    eq: impl Fn(f32, f32) -> bool,
) -> Result<(), ItemMismatchError> {
    if a.len() != b.len() {
        return Err(ItemMismatchError::new(
            device,
            [],
            format!("{} elements", a.len()),
            format!("{} elements", b.len()),
        ));
    }
    for (flat, (va, vb)) in a.iter().zip(b).enumerate() {
        if !eq(*va, *vb) {
            return Err(ItemMismatchError::new(
                device,
                index_of(shape, flat),
                va,
                vb,
            ));
        }
    }
    Ok(())
}

/// `|a - b| <= tol`.
pub fn approx_eq(
    device: &str,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
    tol: f32,
) -> Result<(), ItemMismatchError> {
    eq_with(device, shape, a, b, |va, vb| (va - vb).abs() <= tol)
}

/// Bit-for-bit equality, for indices and for the QAT export path.
pub fn exact_eq(
    device: &str,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
) -> Result<(), ItemMismatchError> {
    eq_with(device, shape, a, b, |va, vb| va == vb)
}

/// `|a - b| <= rel * max(|a|, |b|, f32::MIN_POSITIVE)`.
///
/// Used where reduction outputs grow with the reduced extent and an absolute
/// tolerance stops meaning anything: a sum of 2,025 values of magnitude 5 has
/// roundoff proportional to the result, not to 1.
pub fn relative_eq(
    device: &str,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
    rel: f32,
) -> Result<(), ItemMismatchError> {
    eq_with(device, shape, a, b, |va, vb| {
        let scale = va.abs().max(vb.abs()).max(f32::MIN_POSITIVE);
        (va - vb).abs() <= rel * scale
    })
}

/// Either tolerance passes. For outputs near zero at some inputs and large at
/// others.
pub fn approx_or_relative_eq(
    device: &str,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
    abs: f32,
    rel: f32,
) -> Result<(), ItemMismatchError> {
    eq_with(device, shape, a, b, |va, vb| {
        let diff = (va - vb).abs();
        let scale = va.abs().max(vb.abs()).max(f32::MIN_POSITIVE);
        diff <= abs || diff <= rel * scale
    })
}

/// A comparator, produced by the `*_compare` factories below.
///
/// The reference returns `CompareFut<'a, E>` because its readback is async;
/// `fusor2::Tensor::to_vec_f32` is blocking (readback is one of exactly three
/// host syncs, and a case already sits on the calling thread), so these are
/// plain closures. Nothing else about their role changes.
pub type Comparator = Box<dyn Fn(&str, &[usize], &[f32], &[f32]) -> Result<(), ItemMismatchError>>;

pub fn exact_compare() -> Comparator {
    Box::new(exact_eq)
}

pub fn approx_or_relative_compare(abs: f32, rel: f32) -> Comparator {
    Box::new(move |d, s, a, b| approx_or_relative_eq(d, s, a, b, abs, rel))
}

/// The comparator [`DTYPE_TOL`] prescribes for `dtype`.
pub fn compare_for(dtype: Dtype) -> Comparator {
    let (abs, rel) = tol_for(dtype);
    if abs == 0.0 && rel == 0.0 {
        exact_compare()
    } else {
        approx_or_relative_compare(abs, rel)
    }
}

// ---------------------------------------------------------------------------
// The two names `lib.rs` re-exports
// ---------------------------------------------------------------------------

/// Elementwise `|a - b| <= atol + rtol * |b|`.
pub fn allclose(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs())
}

/// [`allclose`], reporting the worst offender on failure.
pub fn assert_close(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    let worst = a
        .iter()
        .zip(b)
        .enumerate()
        .map(|(i, (x, y))| {
            let slack = (x - y).abs() - (atol + rtol * y.abs());
            (i, *x, *y, slack)
        })
        .max_by(|l, r| l.3.total_cmp(&r.3));
    match worst {
        Some((i, x, y, slack)) if slack > 0.0 => Err(format!(
            "worst mismatch at {i}: {x} vs {y} (over tolerance by {slack:e}; \
             atol={atol:e} rtol={rtol:e})"
        )),
        _ => Ok(()),
    }
}

/// Exact byte equality, for the MSQ1 export gate.
pub fn assert_bytes_eq(a: &[u8], b: &[u8]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {} bytes", a.len(), b.len()));
    }
    match a.iter().zip(b).position(|(x, y)| x != y) {
        None => Ok(()),
        Some(at) => {
            let differing = a.iter().zip(b).filter(|(x, y)| x != y).count();
            Err(format!(
                "first byte difference at {at}: 0x{:02x} vs 0x{:02x} \
                 ({differing} of {} bytes differ)",
                a[at],
                b[at],
                a.len()
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Backend parity
// ---------------------------------------------------------------------------

/// Every case that produces a tensor runs on both sessions and diffs through
/// here. `tol` is `(absolute, relative)`, normally straight from [`tol_for`].
pub fn assert_backend_parity(
    cpu: &[f32],
    gpu: &[f32],
    tol: (f32, f32),
) -> Result<(), ItemMismatchError> {
    let shape = [cpu.len()];
    approx_or_relative_eq("cpu-vs-gpu", &shape, cpu, gpu, tol.0, tol.1)
}

// ---------------------------------------------------------------------------
// Finite differences
// ---------------------------------------------------------------------------

/// The **starting** central-difference step. `1e-2`, not `1e-6`: the graph
/// evaluates in f32 and a smaller step is swamped by cancellation, which is
/// why the reference's autograd tests use this value.
///
/// It is a starting step because a fixed one is not a valid oracle for a
/// piecewise-smooth function — see [`finite_difference_gradient`].
pub const FD_EPSILON: f32 = 1e-2;

/// Factor the step shrinks by when the two one-sided slopes disagree.
pub const FD_SHRINK: f32 = 4.0;

/// How many times the step may shrink. Eight shrinks take `1e-2` down to
/// `1.5e-7`, past which the difference of two f32 losses is all cancellation.
pub const FD_REFINEMENTS: usize = 8;

/// Absolute slack allowed between the analytic and the numeric gradient.
pub const FD_ABS_TOL: f32 = 1e-2;
/// Relative slack, against the numeric gradient's own magnitude.
pub const FD_REL_TOL: f32 = 1e-2;

/// A scalar loss as a function of one input tensor's flat values.
pub type LossFn<'a> = &'a mut dyn FnMut(&[f32]) -> Result<f32, CaseError>;

/// One element's numeric derivative: a central difference whose step shrinks
/// until the two one-sided slopes agree.
///
/// `base` is `loss(data)` at the unperturbed point — the same value for every
/// element, so the caller evaluates it once.
fn refined_partial(
    base: f32,
    slot: usize,
    probe: &mut [f32],
    loss: LossFn<'_>,
) -> Result<f32, CaseError> {
    let original = probe[slot];
    let mut eps = FD_EPSILON;
    // The smallest disagreement seen and the central difference that came
    // with it. Kept so that a shrink which makes things *worse* — the
    // f32-noise regime, where the two slopes diverge as `eps` falls because
    // the numerator is cancellation — cannot be what is returned. There the
    // first and largest step wins, which is the fixed-step central difference
    // this replaced.
    let mut best: Option<(f32, f32)> = None;
    for _ in 0..=FD_REFINEMENTS {
        probe[slot] = original + eps;
        let h_up = probe[slot] - original;
        let up = loss(probe)?;
        probe[slot] = original - eps;
        let h_down = original - probe[slot];
        let down = loss(probe)?;
        probe[slot] = original;
        // `eps` has fallen below an ulp of `original`, so there is no step
        // left to take. On a later try the previous candidate stands; on the
        // first, `FD_EPSILON` itself vanished into the value and no finite
        // difference exists at this magnitude. Say so rather than report the
        // 0/0 the fixed-step form reported.
        if h_up == 0.0 || h_down == 0.0 {
            if best.is_none() {
                return Err(format!(
                    "no finite difference at element {slot}: FD_EPSILON ({FD_EPSILON:e}) \
                     is below one ulp of {original:e}"
                )
                .into());
            }
            break;
        }
        let slope_up = (up - base) / h_up;
        let slope_down = (base - down) / h_down;
        // Their average, and the plain central difference when the two half
        // steps are the equal ones f32 usually gives.
        let central = (up - down) / (h_up + h_down);
        // The resolution floor. The loss's own ulp divided by the step is the
        // smallest slope this eps can distinguish from zero; once that
        // exceeds `FD_ABS_TOL`, nothing measured at this step — agreement
        // included — is evidence. Shrinking past the floor is how a
        // sum-of-log-softmax loss of magnitude ~400 "converged" on a
        // quantization artifact: both perturbed losses rounded to the same
        // neighbour of `base`, the two slopes agreed on the same wrong
        // number, and the loop returned it as a derivative. The best
        // resolvable estimate is the honest answer; when even the first and
        // largest step is under the floor, its central is all there is.
        let quantum = base.abs().max(f32::MIN_POSITIVE) * f32::EPSILON;
        if quantum > FD_ABS_TOL * (h_up + h_down) {
            return Ok(best.map_or(central, |(_, c)| c));
        }
        let disagreement = (slope_up - slope_down).abs();
        let allowed = FD_ABS_TOL + FD_REL_TOL * slope_up.abs().max(slope_down.abs());
        if disagreement <= allowed {
            return Ok(central);
        }
        match best {
            Some((seen, _)) if seen <= disagreement => {}
            _ => best = Some((disagreement, central)),
        }
        eps /= FD_SHRINK;
    }
    // The refinements ran out with the slopes still apart. The least-bad
    // estimate is the honest one to hand back; the assertion, not this, is
    // what decides whether the adjoint agrees with it.
    match best {
        Some((_, central)) => Ok(central),
        None => Err(format!("no finite difference at element {slot}").into()),
    }
}

/// The numeric gradient of `loss` with respect to each element of `data`, by
/// central differences with an adaptive step.
///
/// A **fixed** step is not a valid oracle for a piecewise-smooth loss. Every
/// adjoint this suite checks against — `max`, `min`, `rem`, `abs`, `relu`,
/// argmin-style reductions — is smooth only away from a kink or a jump, and a
/// two-sided quotient taken at `FD_EPSILON` straddles one whenever the sample
/// lands within `FD_EPSILON` of it. It then reports a chord across the two
/// pieces, which is not a derivative of either: `max(x, 0.1)` sampled at
/// `x = 0.0953` gives `0.2626` where every subgradient is `0`, and
/// `x % 0.5` sampled just under a multiple gives `-24` where the derivative
/// is `1` on both sides of the jump.
///
/// So the step adapts. The two one-sided slopes are computed as well as the
/// central one; while they disagree by more than the tolerance the comparison
/// itself uses, the step shrinks by [`FD_SHRINK`] and the element is
/// re-sampled, up to [`FD_REFINEMENTS`] times. The criterion is exactly "the
/// derivative is well defined across this step to the accuracy the assertion
/// demands", so where it already holds — the common case — this costs what
/// the fixed form cost. It converges either way: across a kink as soon as
/// both samples land on one piece, and under curvature at `FD_SHRINK` per
/// try, since a smooth loss's disagreement is `eps * |f''|`.
///
/// This is a *strengthening*: the oracle it converges to is tighter than the
/// fixed one, and every finite-difference case in the suite is checked
/// against it.
///
/// `loss` is re-evaluated `2 * data.len() + 1` times (more where the step
/// refines), so this only ever runs at the small shapes the backward matrix
/// uses.
pub fn finite_difference_gradient(
    shape: &[usize],
    data: &[f32],
    loss: LossFn<'_>,
) -> Result<Vec<f32>, CaseError> {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
        return Err(format!(
            "finite differences over shape {shape:?} ({expected} elements) but {} values",
            data.len()
        )
        .into());
    }
    let mut probe = data.to_vec();
    let base = loss(&probe)?;
    let mut grad = vec![0.0f32; data.len()];
    for i in 0..data.len() {
        grad[i] = refined_partial(base, i, &mut probe, &mut *loss)?;
    }
    Ok(grad)
}

/// `|analytic - numeric| < FD_ABS_TOL + FD_REL_TOL * |numeric|`, elementwise.
pub fn assert_gradient_matches_finite_difference(
    analytic: &[f32],
    numeric: &[f32],
) -> Result<(), CaseError> {
    if analytic.len() != numeric.len() {
        return Err(format!(
            "gradient length mismatch: analytic {} vs numeric {}",
            analytic.len(),
            numeric.len()
        )
        .into());
    }
    for (i, (a, n)) in analytic.iter().zip(numeric).enumerate() {
        let slack = FD_ABS_TOL + FD_REL_TOL * n.abs();
        if (a - n).abs() >= slack {
            return Err(format!(
                "gradient {i}: analytic {a} vs finite-difference {n} \
                 (|diff| = {:e}, allowed {slack:e})",
                (a - n).abs()
            )
            .into());
        }
    }
    Ok(())
}

/// Every gradient element is exactly zero, and there *is* a gradient.
///
/// What the twelve comparison cases assert: a comparison must register an
/// adjoint that emits zeros rather than be absent from the table, because the
/// tape validates that every requires-grad parent receives a gradient.
pub fn assert_all_zero(name: &str, grad: &[f32]) -> Result<(), CaseError> {
    if grad.is_empty() {
        return Err(format!("{name}: no gradient was produced at all").into());
    }
    match grad.iter().position(|g| *g != 0.0) {
        None => Ok(()),
        Some(at) => Err(format!("{name}: gradient {at} is {} rather than 0", grad[at]).into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_tolerances_widen_with_narrower_floats() {
        let (f32_abs, _) = tol_for(Dtype::F32);
        let (f16_abs, _) = tol_for(Dtype::F16);
        let (bf16_abs, _) = tol_for(Dtype::BF16);
        assert!(f32_abs < f16_abs && f16_abs < bf16_abs);
        assert!(is_exact(Dtype::U32) && is_exact(Dtype::I32));
        assert!(!is_exact(Dtype::F32));
    }

    #[test]
    fn index_of_is_row_major() {
        assert_eq!(index_of(&[2, 3], 4), vec![1, 1]);
        assert_eq!(index_of(&[2, 3, 4], 23), vec![1, 2, 3]);
        assert_eq!(index_of(&[], 0), Vec::<usize>::new());
    }

    #[test]
    fn eq_with_reports_the_first_offender_by_index() {
        let err = approx_eq(
            "cpu",
            &[2, 2],
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 9.0, 4.0],
            1e-6,
        )
        .unwrap_err();
        assert_eq!(err.position, vec![1, 0]);
        assert!(err.to_string().contains("expected 3"), "{err}");
    }

    #[test]
    fn relative_eq_passes_where_absolute_would_not() {
        let a = [1.0e6];
        let b = [1.0e6 + 1.0];
        assert!(approx_eq("cpu", &[1], &a, &b, 1e-4).is_err());
        assert!(relative_eq("cpu", &[1], &a, &b, 1e-5).is_ok());
    }

    #[test]
    fn approx_or_relative_takes_either_branch() {
        // Near zero the absolute branch carries it.
        assert!(approx_or_relative_eq("cpu", &[1], &[0.0], &[1e-6], 1e-4, 1e-9).is_ok());
        // Far from zero the relative branch does.
        assert!(approx_or_relative_eq("cpu", &[1], &[1e6], &[1e6 + 1.0], 1e-9, 1e-5).is_ok());
    }

    #[test]
    fn length_mismatch_is_a_mismatch_not_a_panic() {
        assert!(exact_eq("cpu", &[3], &[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn allclose_and_assert_close_agree() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0001, 3.0];
        assert!(allclose(&a, &b, 1e-3, 0.0));
        assert!(assert_close(&a, &b, 1e-3, 0.0).is_ok());
        assert!(!allclose(&a, &b, 1e-9, 0.0));
        let err = assert_close(&a, &b, 1e-9, 0.0).unwrap_err();
        assert!(err.contains("worst mismatch at 1"), "{err}");
    }

    #[test]
    fn assert_bytes_eq_names_the_first_difference() {
        assert!(assert_bytes_eq(&[1, 2, 3], &[1, 2, 3]).is_ok());
        let err = assert_bytes_eq(&[1, 2, 3], &[1, 9, 4]).unwrap_err();
        assert!(err.contains("at 1"), "{err}");
        assert!(err.contains("2 of 3 bytes differ"), "{err}");
        assert!(
            assert_bytes_eq(&[1], &[1, 2])
                .unwrap_err()
                .contains("length")
        );
    }

    #[test]
    fn compare_for_picks_exact_for_integers() {
        // 2.000_001 is the smallest decimal literal here that is a *different*
        // f32 from 2.0 (ulp at 2.0 is 2.4e-7), so the exact comparator has
        // something to reject. It is still well inside F32's 1e-4 tolerance.
        assert_ne!(2.000_001f32, 2.0f32);
        let cmp = compare_for(Dtype::U32);
        assert!(cmp("cpu", &[2], &[1.0, 2.0], &[1.0, 2.000_001]).is_err());
        let cmp = compare_for(Dtype::F32);
        assert!(cmp("cpu", &[2], &[1.0, 2.0], &[1.0, 2.000_001]).is_ok());
    }

    #[test]
    fn backend_parity_is_the_dtype_tolerance() {
        let cpu = [1.0f32, 2.0, 3.0];
        let gpu = [1.00001f32, 2.0, 3.0];
        assert!(assert_backend_parity(&cpu, &gpu, tol_for(Dtype::F32)).is_ok());
        assert!(assert_backend_parity(&cpu, &[1.5, 2.0, 3.0], tol_for(Dtype::F32)).is_err());
    }

    #[test]
    fn finite_differences_recover_a_known_derivative() {
        // d/dx sum(x^3) = 3x^2.
        let data = [0.5f32, -1.5, 2.0];
        let numeric = finite_difference_gradient(&[3], &data, &mut |x| {
            Ok(x.iter().map(|v| v * v * v).sum::<f32>())
        })
        .unwrap();
        let analytic: Vec<f32> = data.iter().map(|v| 3.0 * v * v).collect();
        assert_gradient_matches_finite_difference(&analytic, &numeric).unwrap();
    }

    #[test]
    fn finite_differences_catch_a_wrong_adjoint() {
        let data = [0.5f32, -1.5, 2.0];
        let numeric = finite_difference_gradient(&[3], &data, &mut |x| {
            Ok(x.iter().map(|v| v * v * v).sum::<f32>())
        })
        .unwrap();
        // The classic off-by-a-rule adjoint: 2x instead of 3x^2.
        let wrong: Vec<f32> = data.iter().map(|v| 2.0 * v).collect();
        assert!(assert_gradient_matches_finite_difference(&wrong, &numeric).is_err());
    }

    #[test]
    fn finite_differences_handle_a_nonlinear_transcendental() {
        // d/dx sum(tanh(x)) = 1 - tanh(x)^2, the shape the tanh case asserts.
        let data = [0.3f32, -0.9, 1.4, 0.0];
        let numeric = finite_difference_gradient(&[4], &data, &mut |x| {
            Ok(x.iter().map(|v| v.tanh()).sum::<f32>())
        })
        .unwrap();
        let analytic: Vec<f32> = data.iter().map(|v| 1.0 - v.tanh().powi(2)).collect();
        assert_gradient_matches_finite_difference(&analytic, &numeric).unwrap();
    }

    #[test]
    fn finite_differences_reject_a_shape_disagreement() {
        assert!(finite_difference_gradient(&[4], &[1.0, 2.0], &mut |_| Ok(0.0)).is_err());
    }

    #[test]
    fn the_step_shrinks_off_a_kink_rather_than_chording_across_it() {
        // `backward::max_scalar`'s failing sample: the kink is at 0.1 and x
        // sits 0.00475 below it, inside FD_EPSILON.
        let data = [0.095_252_1f32];
        let numeric = finite_difference_gradient(&[1], &data, &mut |x| Ok(x[0].max(0.1))).unwrap();
        // The subgradient of max(x, 0.1) below the kink is 0, and so is every
        // one-sided slope once the step fits under 0.00475.
        assert_gradient_matches_finite_difference(&[0.0], &numeric).unwrap();
        // The fixed step reported 0.2626 here, which is a chord, not a slope.
        let fixed = ((data[0] + FD_EPSILON).max(0.1) - (data[0] - FD_EPSILON).max(0.1))
            / (2.0 * FD_EPSILON);
        assert!(
            (fixed - 0.262_60).abs() < 1e-4,
            "the fixed-step quotient this replaces was {fixed}"
        );
    }

    #[test]
    fn the_step_shrinks_off_a_value_jump() {
        // `elementwise::rem`: x % 0.5 jumps by 0.5 at each multiple, and the
        // sample sits 0.004 under one. The derivative is 1 on both sides.
        let data = [0.496f32];
        let numeric = finite_difference_gradient(&[1], &data, &mut |x| Ok(x[0] % 0.5)).unwrap();
        assert_gradient_matches_finite_difference(&[1.0], &numeric).unwrap();
    }

    #[test]
    fn a_shrunk_step_still_catches_a_wrong_subgradient() {
        // The adaptive step must not turn into "agree with anything": below
        // the kink the only right answer is 0, and 1 — the other side's
        // subgradient — is still rejected.
        let data = [0.094_747_9f32];
        let numeric = finite_difference_gradient(&[1], &data, &mut |x| Ok(x[0].max(0.1))).unwrap();
        assert!(assert_gradient_matches_finite_difference(&[1.0], &numeric).is_err());
    }

    #[test]
    fn the_shrink_loop_does_not_run_where_the_slopes_already_agree() {
        // No curvature, no kink: the two one-sided slopes match at the first
        // step, so the adaptive form costs what the fixed one did.
        let data = [0.5f32, -1.5, 2.0];
        let mut calls = 0usize;
        let numeric = finite_difference_gradient(&[3], &data, &mut |x| {
            calls += 1;
            Ok(x.iter().map(|v| 2.0 * v).sum::<f32>())
        })
        .unwrap();
        assert_eq!(calls, 2 * data.len() + 1, "one base plus two per element");
        assert_gradient_matches_finite_difference(&[2.0, 2.0, 2.0], &numeric).unwrap();
    }

    #[test]
    fn curvature_refines_and_terminates() {
        // x^3 at 0.5 and -1.5 has |f''| just over the agreement tolerance, so
        // the step shrinks once. What matters is the bound: refinement must
        // converge rather than walk out to FD_REFINEMENTS.
        let data = [0.5f32, -1.5, 2.0];
        let mut calls = 0usize;
        let numeric = finite_difference_gradient(&[3], &data, &mut |x| {
            calls += 1;
            Ok(x.iter().map(|v| v * v * v).sum::<f32>())
        })
        .unwrap();
        assert!(
            calls <= 2 * data.len() + 1 + 4 * data.len(),
            "{calls} evaluations for three cubic elements is not a converging refinement"
        );
        let analytic: Vec<f32> = data.iter().map(|v| 3.0 * v * v).collect();
        assert_gradient_matches_finite_difference(&analytic, &numeric).unwrap();
    }

    #[test]
    fn zero_gradient_assert_distinguishes_absent_from_zero() {
        assert!(assert_all_zero("eq", &[0.0, 0.0]).is_ok());
        assert!(assert_all_zero("eq", &[0.0, 1.0]).is_err());
        assert!(
            assert_all_zero("eq", &[]).is_err(),
            "an absent gradient is not a zero gradient"
        );
    }
}
