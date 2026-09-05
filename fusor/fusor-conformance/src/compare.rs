//! Numeric comparison, per-dtype tolerance, finite differences and backend
//! parity.
//!
//! A comparison takes a flat slice plus the shape and reports a
//! multi-dimensional index on failure.

use std::fmt::{self, Debug, Display};

use fusor::Dtype;

use crate::harness::CaseError;

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

/// The starting central-difference step. `1e-2`, not `1e-6`: the graph
/// evaluates in f32 and a smaller step is swamped by cancellation. The step
/// adapts from here — see [`finite_difference_gradient`].
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
    // The smallest disagreement seen and its central difference, so a shrink
    // into the f32-noise regime (slopes diverging as cancellation grows)
    // cannot be what is returned.
    let mut best: Option<(f32, f32)> = None;
    for _ in 0..=FD_REFINEMENTS {
        probe[slot] = original + eps;
        let h_up = probe[slot] - original;
        let up = loss(probe)?;
        probe[slot] = original - eps;
        let h_down = original - probe[slot];
        let down = loss(probe)?;
        probe[slot] = original;
        // `eps` has fallen below an ulp of `original`: no step left to take.
        // On a later try the previous candidate stands; on the first, no
        // finite difference exists at this magnitude.
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
        let central = (up - down) / (h_up + h_down);
        // The resolution floor: the loss's own ulp divided by the step is the
        // smallest slope this eps can distinguish from zero. Once that
        // exceeds `FD_ABS_TOL`, nothing measured at this step — agreement
        // included — is evidence; shrinking further only lets both perturbed
        // losses round to the same neighbour of `base` and agree on a
        // quantization artifact. Return the best resolvable estimate instead.
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
    // The refinements ran out with the slopes still apart; hand back the
    // least-bad estimate and let the assertion decide.
    match best {
        Some((_, central)) => Ok(central),
        None => Err(format!("no finite difference at element {slot}").into()),
    }
}

/// The numeric gradient of `loss` with respect to each element of `data`, by
/// central differences with an adaptive step.
///
/// The step adapts because a fixed one straddles kinks and jumps in
/// piecewise-smooth losses (`max`, `rem`, `relu`, ...) and reports a chord
/// that is not a derivative of either piece. While the two one-sided slopes
/// disagree by more than the assertion tolerance, the step shrinks by
/// [`FD_SHRINK`], up to [`FD_REFINEMENTS`] times.
///
/// `loss` is re-evaluated `2 * data.len() + 1` times (more where the step
/// refines), so this only ever runs at small shapes.
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
    for (i, g) in grad.iter_mut().enumerate() {
        *g = refined_partial(base, i, &mut probe, &mut *loss)?;
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
/// A comparison must register an adjoint that emits zeros rather than be
/// absent from the table: the tape validates that every requires-grad parent
/// receives a gradient.
pub fn assert_all_zero(name: &str, grad: &[f32]) -> Result<(), CaseError> {
    if grad.is_empty() {
        return Err(format!("{name}: no gradient was produced at all").into());
    }
    match grad.iter().position(|g| *g != 0.0) {
        None => Ok(()),
        Some(at) => Err(format!("{name}: gradient {at} is {} rather than 0", grad[at]).into()),
    }
}
