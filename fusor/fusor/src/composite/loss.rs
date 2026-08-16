//! Losses. `softmax_cross_entropy` and the distillation loss are plain taped
//! chains; `softplus_bce_adjoint` rewrites the backward to the single-sigmoid
//! form.
//!
//! Nothing here is a kernel and nothing here declares an adjoint: every loss
//! is a composition of ops that already carry their own backward.

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
/// The max-subtracted `log_softmax` form is the numerically stable one, and
/// its adjoint composes to `softmax - targets`.
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

/// `mean_rows sum_classes [w * softplus(x) - x * z]`, the folded one-vs-all
/// BCE the trainer optimizes.
///
/// `rows` is the batch size the mean is taken over; it is a parameter rather
/// than `logits.dim(0)` so a padded batch still divides by the live row count.
fn folded_bce_loss(
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
/// One-vs-all: the teacher's
/// per-class probability at `temperature` is `sigmoid(teacher / T)`, and the
/// student pays the folded BCE loss against it at the same temperature. The
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
    a.sub_(b)?.sqr()?.flatten_all()?.mean(0)
}
