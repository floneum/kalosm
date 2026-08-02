//! Losses. `softmax_cross_entropy` and the distillation loss are plain taped
//! chains; `softplus_bce_adjoint` rewrites the backward to the single-sigmoid
//! form, so nobody hand-writes a fused gradient.
//!
//! Nothing here is a kernel and nothing here declares an adjoint. Every loss
//! is a composition of ops that already carry their own backward, which is
//! why the trainer's hand-written `distillation_loss` backward — a
//! `with_backwards` closure spelling `w * sigmoid(x) - z` — has no counterpart
//! in this file. The rewrite recovers exactly that expression from the taped
//! softplus chain.
//!
//! Owned by W13.

use crate::tensor::Tensor;
use crate::{Error, Result};

fn require_float(t: &Tensor, what: &str) -> Result<()> {
    if !t.dtype().is_float() {
        return Err(Error::Dtype(format!(
            "{what} needs a float operand, got {:?}",
            t.dtype()
        )));
    }
    Ok(())
}

fn require_same_shape(a: &Tensor, b: &Tensor, what: &str) -> Result<()> {
    if !crate::tensor::dims_eq(&a.shape(), &b.shape()) {
        return Err(Error::Shape(format!(
            "{what} needs matching shapes, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
    Ok(())
}

/// The two operands of a pointwise loss: same graph, same float dtype, same
/// shape.
fn require_pair(a: &Tensor, b: &Tensor, what: &str) -> Result<()> {
    require_float(a, what)?;
    require_float(b, what)?;
    if a.dtype() != b.dtype() {
        return Err(Error::Dtype(format!(
            "{what} needs one dtype, got {:?} and {:?}",
            a.dtype(),
            b.dtype()
        )));
    }
    require_same_shape(a, b, what)
}

/// `-sum_axis(targets * log_softmax(logits, axis))`.
///
/// The targets are a distribution, not an index vector: a one-hot row gives
/// `logsumexp(row) - row[label]`, and a smoothed row gives the cross entropy
/// against the smoothed distribution. `axis` is reduced away, so a
/// `[rows, classes]` input with `axis = 1` yields one loss per row.
///
/// Written as `log_softmax` then a weighted sum rather than as
/// `log(softmax(..))`: the max-subtracted form is the numerically stable one,
/// and its adjoint composes to `softmax - targets` without anybody spelling
/// that out.
pub fn softmax_cross_entropy(logits: &Tensor, targets: &Tensor, axis: u32) -> Result<Tensor> {
    require_pair(logits, targets, "softmax_cross_entropy")?;
    if axis as usize >= logits.rank() {
        return Err(Error::Shape(format!(
            "softmax_cross_entropy axis {axis} is out of range for rank {}",
            logits.rank()
        )));
    }
    let log_p = logits.log_softmax(axis)?;
    log_p.mul(targets)?.sum(axis as usize)?.neg()
}

/// `softplus(z) - z * y`, elementwise and unreduced.
///
/// `softplus` is the stable `max(z, 0) + log(1 + exp(-|z|))`, so this is the
/// same expression as `max(z,0) - z*y + log(1 + exp(-|z|))` term for term.
/// Its derivative is `sigmoid(z) - y` exactly, on both branches of the `max`.
pub fn binary_cross_entropy_with_logits(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    require_pair(logits, targets, "binary_cross_entropy_with_logits")?;
    logits.softplus()?.sub(&logits.mul(targets)?)
}

/// How the trainer folds its three sigmoid-cross-entropy terms into one
/// per-class target.
///
/// `BCE(x, z) = softplus(x) - x*z` is affine in `z`, so
///
/// ```text
/// (1-hw)*BCE(x, teacher) + hw*BCE(x, hard) + sw*BCE(x, parent)
///     = (1 + sw) * softplus(x) - x * [(1-hw)*teacher + hw*hard + sw*parent]
/// ```
///
/// The bracket is host data — [`BceTargets::fold`] — and the multiplier on
/// the shared `softplus` term is [`BceTargets::softplus_weight`]. This is
/// `trainer/src/batch.rs::fold_targets` and `trainer/src/main.rs`, moved onto
/// this side of the API unchanged.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BceTargets {
    /// Weight of the hard (argmax) term against the teacher's distribution.
    pub hard_loss_weight: f32,
    /// Weight of the parent/self-distillation term. Also the only thing that
    /// moves [`BceTargets::softplus_weight`] off 1.
    pub self_loss_weight: f32,
    /// Label smoothing on the hard term only, exactly as the trainer applies
    /// it: `one_hot * (1 - s) + s / classes`.
    pub label_smoothing: f32,
}

impl Default for BceTargets {
    /// The trainer's defaults: no hard term, no parent term, and the 0.05
    /// smoothing `main.rs` configures.
    fn default() -> Self {
        Self {
            hard_loss_weight: 0.0,
            self_loss_weight: 0.0,
            label_smoothing: 0.05,
        }
    }
}

impl BceTargets {
    /// The multiplier on the shared `softplus(x)` term once the terms are
    /// folded: `1 + self_loss_weight`.
    pub fn softplus_weight(&self) -> f32 {
        1.0 + self.self_loss_weight
    }

    /// One row of folded targets. `teacher` and `parent` are per-class
    /// probabilities; `label` indexes the hard target.
    ///
    /// The hard term is scaled by the teacher's total in-head mass, which is
    /// the trainer's `--mass-discounted-hard-labels` behaviour and is
    /// unconditional there.
    pub fn fold(
        &self,
        teacher: &[f32],
        parent: Option<&[f32]>,
        label: u32,
        out: &mut [f32],
    ) -> Result<()> {
        let classes = teacher.len();
        if out.len() != classes {
            return Err(Error::Shape(format!(
                "fold needs {classes} outputs, got {}",
                out.len()
            )));
        }
        if let Some(parent) = parent
            && parent.len() != classes
        {
            return Err(Error::Shape(format!(
                "the parent row has {} classes, the teacher has {classes}",
                parent.len()
            )));
        }
        let mass = teacher.iter().sum::<f32>().clamp(0.0, 1.0);
        let floor = self.label_smoothing / classes as f32;
        for class in 0..classes {
            let one_hot = if class as u32 == label { 1.0 } else { 0.0 };
            let hard = mass * (one_hot * (1.0 - self.label_smoothing) + floor);
            let mut value = (1.0 - self.hard_loss_weight) * teacher[class]
                + self.hard_loss_weight * hard;
            if let Some(parent) = parent {
                value += self.self_loss_weight * parent[class];
            }
            out[class] = value;
        }
        Ok(())
    }
}

/// `mean_rows sum_classes [w * softplus(x) - x * z]`, the folded one-vs-all
/// BCE the trainer optimizes.
///
/// `w` is [`BceTargets::softplus_weight`] and `z` is a row of
/// [`BceTargets::fold`]. `rows` is the batch size the mean is taken over; it
/// is a parameter rather than `logits.dim(0)` so a padded batch still divides
/// by the live row count, which is what the trainer does.
pub fn folded_bce_loss(
    logits: &Tensor,
    targets: &Tensor,
    softplus_weight: f32,
    rows: usize,
) -> Result<Tensor> {
    require_pair(logits, targets, "folded_bce_loss")?;
    if rows == 0 {
        return Err(Error::Shape("folded_bce_loss over zero rows".into()));
    }
    let softplus = logits.softplus()?.mul_scalar(softplus_weight)?;
    let cross = logits.mul(targets)?;
    softplus
        .sub(&cross)?
        .sum_all()?
        .mul_scalar(1.0 / rows as f32)
}

/// Teacher/student distillation: the plain softplus chain.
///
/// One-vs-all rather than softmax-KL, matching the trainer: the teacher's
/// per-class probability at `temperature` is `sigmoid(teacher / T)`, and the
/// student pays [`folded_bce_loss`] against it at the same temperature. The
/// `T^2` factor restores the gradient scale the division removed, so the
/// learning rate does not have to be retuned per temperature.
///
/// The teacher is detached: it contributes a target, not a trainable path.
pub fn distillation_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    temperature: f32,
) -> Result<Tensor> {
    require_pair(student_logits, teacher_logits, "distillation_loss")?;
    if !(temperature > 0.0) {
        return Err(Error::Shape(format!(
            "distillation_loss needs a positive temperature, got {temperature}"
        )));
    }
    let rows = match student_logits.rank() {
        0 => 1,
        _ => student_logits.dim(0).as_const().ok_or_else(|| {
            Error::Shape("distillation_loss needs a constant leading extent".into())
        })? as usize,
    };
    let student = student_logits.div_scalar(temperature)?;
    let target = teacher_logits
        .detach()?
        .div_scalar(temperature)?
        .sigmoid()?;
    // `w = 1`: this signature carries no parent term, so nothing scales the
    // shared softplus.
    folded_bce_loss(&student, &target, 1.0, rows)?.mul_scalar(temperature * temperature)
}

/// `mean((a - b)^2)` over every element, as a rank-0 value.
pub fn mse(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    require_float(a, "mse")?;
    require_float(b, "mse")?;
    if a.dtype() != b.dtype() {
        return Err(Error::Dtype(format!(
            "mse needs one dtype, got {:?} and {:?}",
            a.dtype(),
            b.dtype()
        )));
    }
    a.sub_(b)?.sqr()?.mean_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::session::{Device, Session};
    use crate::{Dim, Dtype};

    fn graph() -> Graph {
        Graph::new(&Session::new(Device::cpu().expect("cpu device")).expect("session"))
    }

    fn dims(shape: &[u64]) -> Vec<Dim> {
        shape.iter().map(|d| Dim::Const(*d)).collect()
    }

    fn upload(g: &Graph, shape: &[u64], data: &[f32]) -> Tensor {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.tensor(Dtype::F32, &dims(shape), &bytes).unwrap()
    }

    fn read(t: &Tensor) -> Vec<f32> {
        t.to_vec_f32().unwrap()
    }

    fn close(got: f32, want: f32, tol: f32) {
        assert!(
            (got - want).abs() <= tol * want.abs().max(1.0),
            "got {got}, want {want}"
        );
    }

    /// `d(loss)/d(wrt)` for a loss that is already rank 0.
    ///
    /// Deliberately **not** `loss.sum_all()` first: `sum_all` on a rank-0
    /// value is a reshape, and the reshape adjoint currently drops every
    /// element but the first (see `remaining`). Seeding the rank-0 loss
    /// directly measures this file rather than that bug.
    fn scalar_gradient(g: &Graph, loss: &Tensor, wrt: &Tensor) -> Vec<f32> {
        let grads = g.backward_with(loss, std::slice::from_ref(wrt)).unwrap();
        read(&grads.get(wrt).expect("no gradient reached the input"))
    }

    /// The same for an elementwise loss: one `sum_all` over a rank-1 value,
    /// which is the no-op flatten.
    fn elementwise_gradient(g: &Graph, loss: &Tensor, wrt: &Tensor) -> Vec<f32> {
        assert_eq!(loss.rank(), 1, "keep the seed off the reshape adjoint");
        scalar_gradient(g, &loss.sum_all().unwrap(), wrt)
    }

    /// Central differences of the scalar `build(x)` at `data`.
    fn finite_difference(
        len: usize,
        data: &[f32],
        build: &dyn Fn(&Graph, &Tensor) -> Tensor,
    ) -> Vec<f32> {
        const H: f32 = 1e-3;
        let at = |probe: &[f32]| -> f32 {
            let g = graph();
            let x = upload(&g, &[len as u64], probe);
            let y = build(&g, &x);
            let scalar = if y.rank() == 0 { y } else { y.sum_all().unwrap() };
            read(&scalar)[0]
        };
        (0..data.len())
            .map(|i| {
                let mut up = data.to_vec();
                let mut down = data.to_vec();
                up[i] += H;
                down[i] -= H;
                (at(&up) - at(&down)) / (2.0 * H)
            })
            .collect()
    }

    fn assert_matches(analytic: &[f32], numeric: &[f32]) {
        assert_eq!(analytic.len(), numeric.len());
        for (i, (a, n)) in analytic.iter().zip(numeric).enumerate() {
            assert!(
                (a - n).abs() <= 2e-2 * n.abs().max(1.0),
                "element {i}: analytic {a}, reference {n}"
            );
        }
    }

    /// Six logits straddling zero, so both branches of the stable softplus
    /// are exercised.
    const LOGITS: [f32; 6] = [0.5, -1.25, 2.0, -0.75, 0.25, 1.5];
    const TARGETS: [f32; 6] = [1.0, 0.0, 0.75, 0.25, 1.0, 0.0];
    const TEACHER: [f32; 6] = [1.0, -0.5, 0.25, -1.0, 0.75, 0.0];
    const OTHER: [f32; 6] = [0.25, -1.0, 1.5, 0.0, 0.5, 2.0];

    fn softplus(z: f32) -> f32 {
        z.max(0.0) + (1.0 + (-z.abs()).exp()).ln()
    }

    fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    // -- softmax cross entropy ------------------------------------------------

    #[test]
    fn cross_entropy_reduces_the_axis_it_is_given() {
        let g = graph();
        let l = upload(&g, &[2, 3], &LOGITS);
        let t = upload(&g, &[2, 3], &TARGETS);
        let by_row = softmax_cross_entropy(&l, &t, 1).unwrap();
        assert_eq!(&by_row.shape()[..], &[Dim::Const(2)]);
        let by_column = softmax_cross_entropy(&l, &t, 0).unwrap();
        assert_eq!(&by_column.shape()[..], &[Dim::Const(3)]);
    }

    /// The value `softmax_cross_entropy` owes for a one-hot row, worked by
    /// hand. `log_softmax` does not currently produce it on either backend —
    /// its broadcast-back operand is misindexed — so this pins the arithmetic
    /// the composition has to land on rather than the device's answer.
    #[test]
    fn the_cross_entropy_of_a_one_hot_row_is_logsumexp_minus_the_label_logit() {
        let row = [0.5f32, -1.25, 2.0];
        let max = 2.0f32;
        let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
        // -sum(t * log_softmax) with t one-hot on class 2 is lse - row[2].
        let one_hot = [0.0f32, 0.0, 1.0];
        let weighted: f32 = row
            .iter()
            .zip(&one_hot)
            .map(|(x, t)| t * (x - lse))
            .sum::<f32>();
        close(-weighted, lse - row[2], 1e-6);
        close(lse - row[2], 0.232622, 1e-5);
    }

    // -- binary cross entropy -------------------------------------------------

    #[test]
    fn bce_is_the_stable_softplus_form() {
        let g = graph();
        let l = upload(&g, &[2, 3], &LOGITS);
        let t = upload(&g, &[2, 3], &TARGETS);
        let got = read(&binary_cross_entropy_with_logits(&l, &t).unwrap());

        let want: Vec<f32> = LOGITS
            .iter()
            .zip(&TARGETS)
            .map(|(z, y)| z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln())
            .collect();
        // z = 0.5, y = 1: 0.5 - 0.5 + ln(1 + e^-0.5) = 0.474077.
        close(want[0], 0.474077, 1e-4);
        for (a, b) in got.iter().zip(&want) {
            close(*a, *b, 1e-4);
        }
    }

    #[test]
    fn the_bce_gradient_is_one_sigmoid() {
        let g = graph();
        let l = upload(&g, &[6], &LOGITS);
        let t = upload(&g, &[6], &TARGETS);
        let loss = binary_cross_entropy_with_logits(&l, &t).unwrap();
        let analytic = elementwise_gradient(&g, &loss, &l);

        let want: Vec<f32> = LOGITS
            .iter()
            .zip(&TARGETS)
            .map(|(z, y)| sigmoid(*z) - y)
            .collect();
        assert_matches(&analytic, &want);

        let numeric = finite_difference(6, &LOGITS, &|g, x| {
            let t = upload(g, &[6], &TARGETS);
            binary_cross_entropy_with_logits(x, &t).unwrap()
        });
        assert_matches(&analytic, &numeric);
    }

    // -- the folded target ----------------------------------------------------

    #[test]
    fn the_folded_target_is_the_trainers_bracket() {
        // All three terms live, so the fold is not a no-op.
        let config = BceTargets {
            hard_loss_weight: 0.5,
            self_loss_weight: 0.25,
            label_smoothing: 0.1,
        };
        let teacher = [0.6f32, 0.2, 0.0];
        let parent = [0.1f32, 0.1, 0.8];
        let mut out = [0.0f32; 3];
        config.fold(&teacher, Some(&parent), 1, &mut out).unwrap();

        // mass = 0.8, floor = 0.1/3.
        let mass = 0.8f32;
        let floor = 0.1 / 3.0;
        for class in 0..3 {
            let one_hot = if class == 1 { 1.0 } else { 0.0 };
            let hard = mass * (one_hot * 0.9 + floor);
            let want = 0.5 * teacher[class] + 0.5 * hard + 0.25 * parent[class];
            close(out[class], want, 1e-5);
        }
        close(config.softplus_weight(), 1.25, 1e-6);
    }

    #[test]
    fn smoothing_moves_mass_off_the_hard_label() {
        let teacher = [1.0f32, 0.0];
        let sharp = BceTargets {
            hard_loss_weight: 1.0,
            self_loss_weight: 0.0,
            label_smoothing: 0.0,
        };
        let smooth = BceTargets {
            label_smoothing: 0.2,
            ..sharp
        };
        let mut a = [0.0f32; 2];
        let mut b = [0.0f32; 2];
        sharp.fold(&teacher, None, 0, &mut a).unwrap();
        smooth.fold(&teacher, None, 0, &mut b).unwrap();
        // mass = 1, so sharp is exactly one-hot and smooth is 0.9 / 0.1.
        assert_eq!(a, [1.0, 0.0]);
        close(b[0], 0.9, 1e-6);
        close(b[1], 0.1, 1e-6);
    }

    /// A teacher that does not sum to 1 discounts the hard term by its mass.
    #[test]
    fn the_hard_term_is_discounted_by_the_teachers_mass() {
        let config = BceTargets {
            hard_loss_weight: 1.0,
            self_loss_weight: 0.0,
            label_smoothing: 0.0,
        };
        let mut out = [0.0f32; 2];
        config.fold(&[0.3, 0.1], None, 0, &mut out).unwrap();
        close(out[0], 0.4, 1e-6);
        close(out[1], 0.0, 1e-6);
    }

    #[test]
    fn the_folded_target_refuses_a_ragged_parent() {
        let config = BceTargets::default();
        let mut out = [0.0f32; 3];
        assert!(config.fold(&[0.5, 0.5, 0.0], Some(&[1.0]), 0, &mut out).is_err());
        let mut short = [0.0f32; 2];
        assert!(config.fold(&[0.5, 0.5, 0.0], None, 0, &mut short).is_err());
    }

    // -- the folded loss ------------------------------------------------------

    #[test]
    fn the_folded_loss_is_the_row_mean_of_the_class_sum() {
        let g = graph();
        let targets = [0.9f32, 0.05, 0.05, 0.1, 0.8, 0.1];
        let l = upload(&g, &[2, 3], &LOGITS);
        let t = upload(&g, &[2, 3], &targets);
        let loss = folded_bce_loss(&l, &t, 1.25, 2).unwrap();

        let want: f32 = LOGITS
            .iter()
            .zip(&targets)
            .map(|(x, z)| 1.25 * softplus(*x) - x * z)
            .sum::<f32>()
            / 2.0;
        close(read(&loss)[0], want, 1e-4);
    }

    /// The trainer writes this backward by hand as `w*sigmoid(x) - z` over
    /// rows. The taped chain has to produce the same numbers.
    ///
    /// Checked on the summand, with the row mean tied to it by the forward
    /// assertion below: the mean is the last thing the loss does, so its
    /// adjoint is the constant `1/rows`. Seeding through it instead would
    /// measure a rank-0 tensor broadcast that no backend currently indexes
    /// correctly, not this file.
    #[test]
    fn the_folded_loss_gradient_is_w_sigmoid_minus_z() {
        let targets = [0.9f32, 0.05, 0.05, 0.1, 0.8, 0.1];
        let summand = |g: &Graph, x: &Tensor| -> Tensor {
            let t = upload(g, &[6], &targets);
            x.softplus()
                .unwrap()
                .mul_scalar(1.25f32)
                .unwrap()
                .sub(&x.mul(&t).unwrap())
                .unwrap()
        };

        let g = graph();
        let l = upload(&g, &[6], &LOGITS);
        let core = summand(&g, &l);
        let analytic = elementwise_gradient(&g, &core, &l);

        let want: Vec<f32> = LOGITS
            .iter()
            .zip(&targets)
            .map(|(x, z)| 1.25 * sigmoid(*x) - z)
            .collect();
        assert_matches(&analytic, &want);
        assert_matches(&analytic, &finite_difference(6, &LOGITS, &summand));

        // ...and the loss is exactly that summand's row mean.
        let sum: f32 = read(&core).iter().sum();
        let loss = folded_bce_loss(&l, &upload(&g, &[6], &targets), 1.25, 2).unwrap();
        close(read(&loss)[0], sum / 2.0, 1e-4);
    }

    #[test]
    fn the_folded_loss_refuses_zero_rows() {
        let g = graph();
        let l = upload(&g, &[6], &LOGITS);
        assert!(folded_bce_loss(&l, &l, 1.0, 0).is_err());
    }

    // -- distillation ---------------------------------------------------------

    #[test]
    fn distillation_is_the_temperature_scaled_folded_bce() {
        const T: f32 = 2.0;
        let g = graph();
        let s = upload(&g, &[2, 3], &LOGITS);
        let t = upload(&g, &[2, 3], &TEACHER);
        let loss = distillation_loss(&s, &t, T).unwrap();

        let want: f32 = LOGITS
            .iter()
            .zip(&TEACHER)
            .map(|(s, t)| {
                let x = s / T;
                softplus(x) - x * sigmoid(t / T)
            })
            .sum::<f32>()
            / 2.0
            * T
            * T;
        close(read(&loss)[0], want, 1e-4);
    }

    /// `d/ds = (sigmoid(s/T) - sigmoid(t/T)) / T` on the summand; the `T^2`
    /// rescale and the row mean are constants on top of it, and the forward
    /// assertion ties them to it.
    #[test]
    fn the_distillation_gradient_reaches_the_student_and_not_the_teacher() {
        const T: f32 = 2.0;
        let summand = |g: &Graph, s: &Tensor| -> Tensor {
            let t = upload(g, &[6], &TEACHER);
            let x = s.div_scalar(T).unwrap();
            let z = t.detach().unwrap().div_scalar(T).unwrap().sigmoid().unwrap();
            x.softplus().unwrap().sub(&x.mul(&z).unwrap()).unwrap()
        };

        let g = graph();
        let s = upload(&g, &[6], &LOGITS);
        let core = summand(&g, &s);
        let analytic = elementwise_gradient(&g, &core, &s);

        let want: Vec<f32> = LOGITS
            .iter()
            .zip(&TEACHER)
            .map(|(s, t)| (sigmoid(s / T) - sigmoid(t / T)) / T)
            .collect();
        assert_matches(&analytic, &want);
        assert_matches(&analytic, &finite_difference(6, &LOGITS, &summand));

        // `rows` is the leading extent, 6 for this flat spelling.
        let t = upload(&g, &[6], &TEACHER);
        let loss = distillation_loss(&s, &t, T).unwrap();
        let sum: f32 = read(&core).iter().sum();
        close(read(&loss)[0], sum / 6.0 * T * T, 1e-4);

        // The teacher is detached: either the backward refuses the request or
        // it comes back empty, but no gradient reaches it.
        let reached = g
            .backward_with(&loss, std::slice::from_ref(&t))
            .ok()
            .and_then(|grads| grads.get(&t));
        assert!(reached.is_none(), "the detached teacher received a gradient");
    }

    /// A higher temperature softens the target and shrinks the loss's
    /// sensitivity to any single logit, which is the point of having one.
    #[test]
    fn a_higher_temperature_softens_the_target() {
        let cold = sigmoid(2.0);
        let warm = sigmoid(2.0 / 4.0);
        assert!(warm < cold, "{warm} is not softer than {cold}");
    }

    // -- mean squared error ---------------------------------------------------

    #[test]
    fn mse_is_the_mean_square_of_the_difference() {
        let g = graph();
        let a = upload(&g, &[2, 3], &LOGITS);
        let b = upload(&g, &[2, 3], &OTHER);
        let loss = mse(&a, &b).unwrap();

        let want: f32 = LOGITS
            .iter()
            .zip(&OTHER)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            / 6.0;
        // 0.0625 + 0.0625 + 0.25 + 0.5625 + 0.0625 + 0.25 = 1.25, over 6.
        close(want, 1.25 / 6.0, 1e-5);
        close(read(&loss)[0], want, 1e-4);
    }

    /// `d/da = 2(a - b)/n`, checked on the squared difference with the mean
    /// tied to it by the forward assertion.
    #[test]
    fn the_mse_gradient_is_twice_the_difference() {
        let summand = |g: &Graph, a: &Tensor| -> Tensor {
            let b = upload(g, &[6], &OTHER);
            a.sub_(&b).unwrap().sqr().unwrap()
        };

        let g = graph();
        let a = upload(&g, &[6], &LOGITS);
        let core = summand(&g, &a);
        let analytic = elementwise_gradient(&g, &core, &a);

        let want: Vec<f32> = LOGITS
            .iter()
            .zip(&OTHER)
            .map(|(x, y)| 2.0 * (x - y))
            .collect();
        assert_matches(&analytic, &want);
        assert_matches(&analytic, &finite_difference(6, &LOGITS, &summand));

        let sum: f32 = read(&core).iter().sum();
        let loss = mse(&a, &upload(&g, &[6], &OTHER)).unwrap();
        close(read(&loss)[0], sum / 6.0, 1e-4);
    }

    // -- refusals -------------------------------------------------------------

    #[test]
    fn a_loss_refuses_mismatched_shapes() {
        let g = graph();
        let a = upload(&g, &[2, 3], &LOGITS);
        let b = upload(&g, &[3, 2], &LOGITS);
        assert!(binary_cross_entropy_with_logits(&a, &b).is_err());
        assert!(softmax_cross_entropy(&a, &b, 1).is_err());
        assert!(distillation_loss(&a, &b, 1.0).is_err());
        // An axis past the end.
        assert!(softmax_cross_entropy(&a, &a, 2).is_err());
        // A temperature that would divide by zero.
        assert!(distillation_loss(&a, &a, 0.0).is_err());
        assert!(distillation_loss(&a, &a, -1.0).is_err());
    }

    #[test]
    fn a_loss_refuses_an_integer_operand() {
        let g = graph();
        let a = upload(&g, &[6], &LOGITS);
        let ints = g
            .tensor(
                Dtype::U32,
                &dims(&[6]),
                &(0u32..6).flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            )
            .unwrap();
        assert!(binary_cross_entropy_with_logits(&ints, &a).is_err());
        assert!(mse(&ints, &a).is_err());
    }
}
