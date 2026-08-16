//! Optimizers. AdamW's `m * lr_f32` produces a `Uniform` read from binding 0,
//! never a baked literal — so a learning-rate schedule does not recompile
//! anything.
//!
//! Bias correction is folded into the step size, weight decay is decoupled and
//! applied to the parameter rather than the gradient, and epsilon is added to
//! `sqrt(v)` rather than to `v`. The two per-step scalars are the only things
//! that change between steps and both are uniforms, so a whole training run
//! compiles one kernel set.

use std::sync::Arc;

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
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u64,
    /// `(m, v)` per parameter, in the order the first `step` saw them.
    state: Vec<Moments>,
    schedule: Option<Schedule>,
}

impl AdamW {
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
            if !Arc::ptr_eq(p.graph(), &graph) || !Arc::ptr_eq(g.graph(), &graph) {
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
            && !Arc::ptr_eq(previous, &graph)
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
        if !Arc::ptr_eq(g.graph(), &graph) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::session::{Backend, Session};
    use crate::{Dim, Dtype};

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().expect("cpu device")).expect("session"))
    }

    fn upload(g: &Graph, shape: &[u64], data: &[f32]) -> Tensor {
        let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.tensor(Dtype::F32, &dims, &bytes).unwrap()
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

    /// The whole update on the host, one scalar at a time. This is the
    /// specification; the tensor path below has to reproduce it.
    struct HostAdam {
        m: f32,
        v: f32,
        t: i32,
    }

    impl HostAdam {
        fn step(&mut self, p: f32, g: f32, lr: f32, wd: f32, eps: f32) -> f32 {
            const B1: f32 = 0.9;
            const B2: f32 = 0.999;
            self.t += 1;
            self.m = B1 * self.m + (1.0 - B1) * g;
            self.v = B2 * self.v + (1.0 - B2) * g * g;
            let alpha = lr * (1.0 - B2.powi(self.t)).sqrt() / (1.0 - B1.powi(self.t));
            let update = self.m * alpha / (self.v.sqrt() + eps);
            (p - p * (lr * wd)) - update
        }
    }

    #[test]
    fn one_step_matches_a_hand_stepped_scalar() {
        const LR: f32 = 0.1;
        let g = graph();
        let p = upload(&g, &[1], &[1.0]);
        let grad = upload(&g, &[1], &[2.0]);

        let mut opt = AdamW::new(LR);
        let out = opt.step(std::slice::from_ref(&p), std::slice::from_ref(&grad)).unwrap();
        let got = read(&out[0])[0];

        // m = 0.2, v = 0.004, alpha = 0.1*sqrt(0.001)/0.1 = 0.0316228,
        // update = 0.2*0.0316228 / (0.0632456 + 1e-8) = 0.1.
        let mut host = HostAdam { m: 0.0, v: 0.0, t: 0 };
        let want = host.step(1.0, 2.0, LR, 0.0, opt.eps);
        close(want, 0.9, 1e-5);
        close(got, want, 1e-5);
        assert_eq!(opt.step, 1);
    }

    /// The step size after bias correction is the learning rate, whatever the
    /// gradient's scale.
    #[test]
    fn the_first_step_is_the_learning_rate_at_any_gradient_scale() {
        const LR: f32 = 0.05;
        for scale in [1e-3f32, 1.0, 1e3] {
            let g = graph();
            let p = upload(&g, &[1], &[0.0]);
            let grad = upload(&g, &[1], &[scale]);
            let mut opt = AdamW::new(LR);
            let out = opt
                .step(std::slice::from_ref(&p), std::slice::from_ref(&grad))
                .unwrap();
            close(read(&out[0])[0], -LR, 1e-3);
        }
    }

    /// Three steps against the host, with the state carried on the graph.
    #[test]
    fn three_steps_track_the_host_recurrence() {
        const LR: f32 = 0.1;
        const WD: f32 = 0.01;
        let grads = [2.0f32, 1.0, -0.5];

        let g = graph();
        let mut opt = AdamW::new(LR);
        opt.weight_decay = WD;
        let mut host = HostAdam { m: 0.0, v: 0.0, t: 0 };
        let mut want = 1.0f32;
        let mut current = upload(&g, &[1], &[1.0]);

        for grad in grads {
            let gt = upload(&g, &[1], &[grad]);
            let out = opt
                .step(std::slice::from_ref(&current), std::slice::from_ref(&gt))
                .unwrap();
            want = host.step(want, grad, LR, WD, opt.eps);
            let got = read(&out[0])[0];
            close(got, want, 2e-4);
            // Feed the realized value forward, as a training loop would.
            current = upload(&g, &[1], &[got]);
        }
        assert_eq!(opt.step, 3);
    }

    /// Decoupled decay reads the parameter, not the gradient: with a zero
    /// gradient the update is exactly `-p * lr * wd`.
    #[test]
    fn the_decay_is_decoupled_from_the_gradient() {
        const LR: f32 = 0.1;
        const WD: f32 = 0.5;
        let g = graph();
        let p = upload(&g, &[2], &[1.0, -2.0]);
        let grad = upload(&g, &[2], &[0.0, 0.0]);
        let mut opt = AdamW::new(LR);
        opt.weight_decay = WD;
        let out = opt
            .step(std::slice::from_ref(&p), std::slice::from_ref(&grad))
            .unwrap();
        let got = read(&out[0]);
        // m = v = 0, so the Adam term is 0/(0+eps) = 0 and only the decay
        // moves: p * (1 - 0.05).
        close(got[0], 0.95, 1e-5);
        close(got[1], -1.9, 1e-5);
    }

    #[test]
    fn step_refuses_mismatched_inputs() {
        let g = graph();
        let p = upload(&g, &[2], &[1.0, 1.0]);
        let grad = upload(&g, &[3], &[1.0, 1.0, 1.0]);
        let mut opt = AdamW::new(0.1);
        assert!(opt.step(&[p.clone()], &[]).is_err());
        assert!(opt.step(&[p.clone()], &[grad]).is_err());
        // A refused call must not consume a step.
        assert_eq!(opt.step, 0);
        // And the state cannot change parameter count mid-run.
        let one = upload(&g, &[2], &[0.0, 0.0]);
        opt.step(&[p.clone()], std::slice::from_ref(&one)).unwrap();
        assert!(opt.step(&[p.clone(), p], &[one.clone(), one]).is_err());
    }

    #[test]
    fn cosine_decay_ramps_then_falls_to_the_floor() {
        const WARMUP: u64 = 10;
        const TOTAL: u64 = 100;
        const PEAK: f32 = 1.0;
        const FLOOR: f32 = 0.1;
        let at = |s| cosine_decay(s, WARMUP, TOTAL, PEAK, FLOOR);

        assert_eq!(at(0), 0.0);
        close(at(5), 0.5, 1e-6);
        close(at(WARMUP), PEAK, 1e-6);
        // Halfway through the decay the cosine is 0.5, so the rate is the
        // midpoint of peak and floor.
        close(at(55), 0.55, 1e-5);
        close(at(TOTAL), FLOOR, 1e-6);
        // Past the end it holds, it does not turn back up.
        close(at(TOTAL + 50), FLOOR, 1e-6);

        let mut previous = at(WARMUP);
        for step in WARMUP + 1..=TOTAL {
            let now = at(step);
            assert!(now <= previous + 1e-6, "rose at {step}: {previous} -> {now}");
            previous = now;
        }
    }

    #[test]
    fn cosine_decay_without_warmup_starts_at_the_peak() {
        close(cosine_decay(0, 0, 10, 2.0, 0.5), 2.0, 1e-6);
        close(cosine_decay(10, 0, 10, 2.0, 0.5), 0.5, 1e-6);
        // A degenerate span cannot divide by zero.
        assert_eq!(cosine_decay(0, 5, 5, 1.0, 0.25), 0.0);
        assert_eq!(cosine_decay(5, 5, 5, 1.0, 0.25), 0.25);
    }

    /// 3,4 and 0,12 have norms 5 and 12, so the global norm is 13.
    #[test]
    fn the_global_norm_is_the_root_of_every_summed_square() {
        let g = graph();
        let a = upload(&g, &[2], &[3.0, 4.0]);
        let b = upload(&g, &[2], &[0.0, 12.0]);
        close(read(&global_norm(&[a, b]).unwrap())[0], 13.0, 1e-5);
    }

    /// Above the cap the factor is `cap / ||g||`; below it, exactly 1.
    #[test]
    fn the_clip_scale_is_the_cap_over_the_norm() {
        let g = graph();
        let a = upload(&g, &[2], &[3.0, 4.0]);
        let b = upload(&g, &[2], &[0.0, 12.0]);
        let pair = [a, b];
        close(read(&clip_scale(&pair, 1.0).unwrap())[0], 1.0 / 13.0, 1e-5);
        close(read(&clip_scale(&pair, 6.5).unwrap())[0], 0.5, 1e-5);
        // 13 is inside a ball of radius 100, so nothing is scaled.
        close(read(&clip_scale(&pair, 100.0).unwrap())[0], 1.0, 1e-6);
    }

    /// One output per gradient, each in its own shape.
    #[test]
    fn clipping_preserves_the_gradient_shapes() {
        let g = graph();
        let a = upload(&g, &[2], &[3.0, 4.0]);
        let b = upload(&g, &[3], &[0.0, 12.0, 5.0]);
        let clipped = clip_global_norm(&[a, b], 1.0).unwrap();
        assert_eq!(clipped.len(), 2);
        assert_eq!(&clipped[0].shape()[..], &[Dim::Const(2)]);
        assert_eq!(&clipped[1].shape()[..], &[Dim::Const(3)]);
    }

    #[test]
    fn clipping_refuses_a_non_positive_cap() {
        let g = graph();
        let a = upload(&g, &[2], &[1.0, 1.0]);
        assert!(clip_global_norm(std::slice::from_ref(&a), 0.0).is_err());
        assert!(clip_global_norm(std::slice::from_ref(&a), -1.0).is_err());
        // An empty list is not an error; there is nothing to scale. Asking
        // for its norm is, because there is no such number.
        assert!(clip_global_norm(&[], 1.0).unwrap().is_empty());
        assert!(global_norm(&[]).is_err());
    }
}
