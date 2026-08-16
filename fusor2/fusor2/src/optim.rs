//! Optimizers. AdamW's `m * lr_f32` produces a `Uniform` read from binding 0,
//! never a baked literal — so a learning-rate schedule does not recompile
//! anything.
//!
//! Bias correction is folded into the step size, weight decay is decoupled and
//! applied to the parameter rather than the gradient, and epsilon is added to
//! `sqrt(v)` rather than to `v`. The two per-step scalars are the only things
//! that change between steps and both are uniforms, so a whole training run
//! compiles one kernel set.

use fusor2_ir::shape::SymId;

use crate::graph::GraphRef;
use crate::tensor::{Scalar, Tensor};
use crate::{Error, Result};

/// The first and second moment of one parameter. Both are graph values: the
/// state never round-trips through the host.
struct Moments {
    m: Tensor,
    v: Tensor,
}

/// The two scalars a schedule moves. Minted once against the graph the first
/// [`AdamW::step`] saw and then rebound each step, so changing the learning
/// rate changes a word in binding 0 and nothing else.
struct Schedule {
    /// `lr * sqrt(1 - beta2^t) / (1 - beta1^t)`.
    alpha: SymId,
    /// `lr * weight_decay`.
    decay: SymId,
}

/// Decoupled-weight-decay Adam.
///
/// The defaults are the Adam paper's; `eps` and `weight_decay` are public
/// fields.
pub struct AdamW {
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Denominator stability term.
    pub eps: f32,
    /// Decoupled weight-decay coefficient.
    pub weight_decay: f32,
    /// Number of completed optimizer steps.
    pub step: u64,
    /// `(m, v)` per parameter, in the order the first `step` saw them.
    state: Vec<Moments>,
    schedule: Option<Schedule>,
}

impl AdamW {
    /// Create AdamW with standard beta and epsilon defaults.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step: 0,
            state: Vec::new(),
            schedule: None,
        }
    }

    /// The graph the state lives on, once there is any.
    fn state_graph(&self) -> Option<&GraphRef> {
        self.state.first().map(|s| s.m.graph())
    }

    /// One update. Returns the new parameter values in the same order.
    ///
    /// `params[i]` and `grads[i]` must agree on shape and dtype, and every
    /// value must come from one graph — the same graph as the previous call's,
    /// because the moments are values on it.
    ///
    /// The returned tensors are unresolved expressions, as everything in this
    /// API is. Resolve them (or read them) before the next `step`: the moments
    /// this call stored are expressions over `grads`, and the next call's
    /// uniforms overwrite this call's.
    pub fn step(&mut self, params: &[Tensor], grads: &[Tensor]) -> Result<Vec<Tensor>> {
        if params.len() != grads.len() {
            return Err(Error::Shape(format!(
                "AdamW::step got {} parameters and {} gradients",
                params.len(),
                grads.len()
            )));
        }
        if params.is_empty() {
            return Ok(Vec::new());
        }
        if !self.state.is_empty() && self.state.len() != params.len() {
            return Err(Error::Shape(format!(
                "AdamW::step holds state for {} parameters but was given {}",
                self.state.len(),
                params.len()
            )));
        }
        let graph = params[0].graph().clone();
        for (i, (p, g)) in params.iter().zip(grads).enumerate() {
            if !GraphRef::ptr_eq(p.graph(), &graph) || !GraphRef::ptr_eq(g.graph(), &graph) {
                return Err(Error::Device(
                    "AdamW::step got operands from two different graphs".into(),
                ));
            }
            if !p.dtype().is_float() {
                return Err(Error::Dtype(format!(
                    "parameter {i} has dtype {:?}; AdamW needs a float parameter",
                    p.dtype()
                )));
            }
            if p.dtype() != g.dtype() {
                return Err(Error::Dtype(format!(
                    "parameter {i} is {:?} but its gradient is {:?}",
                    p.dtype(),
                    g.dtype()
                )));
            }
            if !crate::tensor::dims_eq(&p.shape(), &g.shape()) {
                return Err(Error::Shape(format!(
                    "parameter {i} has shape {:?} but its gradient has {:?}",
                    p.shape(),
                    g.shape()
                )));
            }
        }
        if let Some(previous) = self.state_graph()
            && !GraphRef::ptr_eq(previous, &graph)
        {
            return Err(Error::Device(
                "AdamW state belongs to another graph; the moments are values on it".into(),
            ));
        }
        if !(self.beta1 < 1.0) || !(self.beta2 < 1.0) {
            return Err(Error::Shape(format!(
                "AdamW needs beta1 and beta2 below 1, got {} and {}",
                self.beta1, self.beta2
            )));
        }

        self.step += 1;
        // `powi` saturates rather than wrapping: past 2^31 steps both
        // corrections have long since reached 1.
        let t = self.step.min(i32::MAX as u64) as i32;
        let bias1 = 1.0 - self.beta1.powi(t);
        let bias2 = 1.0 - self.beta2.powi(t);
        // Bias correction folded into the step size, as Keras does.
        let alpha = self.lr * bias2.sqrt() / bias1;
        let decay = self.lr * self.weight_decay;

        let schedule = self.schedule.get_or_insert_with(|| Schedule {
            alpha: graph.fresh_sym(),
            decay: graph.fresh_sym(),
        });
        graph.set_uniform(schedule.alpha, alpha);
        graph.set_uniform(schedule.decay, decay);
        let alpha = Scalar::Uniform(schedule.alpha);
        let decay = Scalar::Uniform(schedule.decay);

        let mut updated = Vec::with_capacity(params.len());
        let mut next = Vec::with_capacity(params.len());
        for (i, (p, g)) in params.iter().zip(grads).enumerate() {
            let previous = self.state.get(i);
            // `m = beta1*m + (1-beta1)*g`, and likewise for `v` over `g^2`.
            // On the first step the carried term is exactly zero.
            let m = match previous {
                Some(s) => s
                    .m
                    .mul_scalar(self.beta1)?
                    .add(&g.mul_scalar(1.0 - self.beta1)?)?,
                None => g.mul_scalar(1.0 - self.beta1)?,
            };
            let squared = g.mul(g)?;
            let v = match previous {
                Some(s) => s
                    .v
                    .mul_scalar(self.beta2)?
                    .add(&squared.mul_scalar(1.0 - self.beta2)?)?,
                None => squared.mul_scalar(1.0 - self.beta2)?,
            };

            // Epsilon sits outside the root: `m*alpha / (sqrt(v) + eps)`.
            let denominator = v.sqrt()?.add_scalar(self.eps)?;
            let update = m.mul_scalar(alpha)?.div(&denominator)?;
            // Decoupled weight decay: the decay reads the parameter, never
            // the gradient, so it does not enter `m` or `v`.
            let decayed = p.sub(&p.mul_scalar(decay)?)?;
            updated.push(decayed.sub(&update)?);
            next.push(Moments { m, v });
        }
        self.state = next;
        Ok(updated)
    }
}

/// Cosine learning-rate decay with linear warmup.
///
/// `step <= warmup` ramps linearly from 0 to `peak`; after that the Keras
/// `CosineDecay` shape `floor + (peak - floor) * 0.5 * (1 + cos(pi * t))`
/// carries it to `floor` at `total` and holds there. With `warmup = 0` the
/// schedule starts at `peak`.
pub fn cosine_decay(step: u64, warmup: u64, total: u64, peak: f32, floor: f32) -> f32 {
    if step < warmup {
        return peak * (step as f32 / warmup as f32);
    }
    // `total` at or below `warmup` leaves no room to decay in.
    let span = total.saturating_sub(warmup);
    if span == 0 {
        return floor;
    }
    let progress = ((step - warmup) as f32 / span as f32).min(1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    floor + (peak - floor) * cosine
}

/// The Kernel norm of every gradient taken together, as a rank-0 value.
///
/// A device reduction: nothing here reads a gradient back to the host, and
/// the trainer logs this number every step.
pub fn global_norm(grads: &[Tensor]) -> Result<Tensor> {
    let Some(first) = grads.first() else {
        return Err(Error::Shape("the global norm of no gradients".into()));
    };
    let graph = first.graph().clone();
    let mut total: Option<Tensor> = None;
    for (i, g) in grads.iter().enumerate() {
        if !GraphRef::ptr_eq(g.graph(), &graph) {
            return Err(Error::Device(
                "global_norm got gradients from two different graphs".into(),
            ));
        }
        if !g.dtype().is_float() {
            return Err(Error::Dtype(format!(
                "gradient {i} has dtype {:?}; the global norm needs floats",
                g.dtype()
            )));
        }
        let part = g.sqr()?.sum_all()?;
        total = Some(match total {
            Some(acc) => acc.add(&part)?,
            None => part,
        });
    }
    total.expect("a non-empty gradient list has a total").sqrt()
}

/// `max_norm / max(||g||, max_norm)`: the shared factor clipping applies.
///
/// Exactly 1 below the threshold and exactly `max_norm / ||g||` above it, so
/// the clip is a no-op inside the ball and lands the norm on the cap outside
/// it.
fn clip_scale(grads: &[Tensor], max_norm: f32) -> Result<Tensor> {
    global_norm(grads)?
        .max_scalar(max_norm)?
        .rdiv_scalar(max_norm)
}

/// Scale every gradient so the global Kernel norm is at most `max_norm`.
///
/// One shared factor, so the direction is preserved. The factor is a rank-0
/// device value and the scaling is a broadcast multiply against it.
pub fn clip_global_norm(grads: &[Tensor], max_norm: f32) -> Result<Vec<Tensor>> {
    if grads.is_empty() {
        return Ok(Vec::new());
    }
    if !(max_norm > 0.0) {
        return Err(Error::Shape(format!(
            "clip_global_norm needs a positive cap, got {max_norm}"
        )));
    }
    let scale = clip_scale(grads, max_norm)?;
    grads.iter().map(|g| g.mul_(&scale)).collect()
}
