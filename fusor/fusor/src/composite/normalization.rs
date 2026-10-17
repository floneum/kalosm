//! Softmax, RMS normalization and layer normalization over Fold and Map.

use fusor_autograd::tape::{GraphTape, TapeExt, accum_dtype};
use fusor_ir::autograd::{Tape, Val};
use fusor_ir::scalar::{BinOp, ScalarExpr, UnOp};
use fusor_ir::shape::Dim;
use fusor_ir::{Error, Result};

use crate::composite::core_op;
use crate::graph::GraphRef;
use crate::tensor::Tensor;

/// The shared prologue of both softmaxes over `axis`: `x - max`, its `exp`,
/// the sum of that (accumulated wide, cast back) and the axis extent.
fn exp_sum(t: &mut GraphTape<'_>, x: Val, axis: u32, what: &str) -> Result<(Val, Val, Val, Dim)> {
    let shape = t.shape_of(x);
    let dtype = t.dtype_of(x);
    let extent = *shape
        .get(axis as usize)
        .ok_or_else(|| Error::Shape(format!("{what} axis {axis} out of range")))?;

    let m = t.fold_binop(BinOp::Max, axis, dtype, x)?;
    let m = t.broadcast_axis(m, axis, extent)?;
    let centered = t.binary(BinOp::Sub, x, m)?;
    let e = t.unary(UnOp::Exp, centered)?;
    let s = t.fold_binop(BinOp::Add, axis, accum_dtype(dtype), e)?;
    let s = t.cast(dtype, s)?;
    Ok((centered, e, s, extent))
}

/// `exp(x - max) / sum(exp(x - max))` over `axis`. Every entry point shares
/// this.
pub(crate) fn softmax_defn(t: &mut GraphTape<'_>, x: Val, axis: u32) -> Result<Val> {
    let (_, e, s, extent) = exp_sum(t, x, axis, "softmax")?;
    let s = t.broadcast_axis(s, axis, extent)?;
    t.binary(BinOp::Div, e, s)
}

/// `x - max - log(sum(exp(x - max)))`.
pub(crate) fn log_softmax_defn(t: &mut GraphTape<'_>, x: Val, axis: u32) -> Result<Val> {
    let (centered, _, s, extent) = exp_sum(t, x, axis, "log_softmax")?;
    let ls = t.unary(UnOp::Log, s)?;
    let ls = t.broadcast_axis(ls, axis, extent)?;
    t.binary(BinOp::Sub, centered, ls)
}

/// `sum(x, axis) / extent`. Errors on a symbolic extent.
fn mean_axis(t: &mut GraphTape<'_>, x: Val, axis: u32) -> Result<Val> {
    let shape = t.shape_of(x);
    let extent = shape
        .get(axis as usize)
        .copied()
        .ok_or_else(|| Error::Shape(format!("mean axis {axis} out of range")))?;
    let n = extent.as_const().ok_or_else(|| {
        Error::Shape(format!(
            "a normalization axis needs a decidable extent, got {extent}"
        ))
    })?;
    let dtype = t.dtype_of(x);
    let acc = accum_dtype(dtype);
    let s = t.fold_binop(BinOp::Add, axis, acc, x)?;
    let s = t.cast(dtype, s)?;
    t.mul_scalar(s, 1.0 / n.max(1) as f32)
}

/// `x`, mean-centred when `remove_mean`, and the mean of its square over
/// `axis`: the biased variance, or the RMS statistic when not centred.
fn centered_variance(
    t: &mut GraphTape<'_>,
    x: Val,
    axis: u32,
    remove_mean: bool,
) -> Result<(Val, Val)> {
    let centered = if remove_mean {
        let extent = t.shape_of(x)[axis as usize];
        let mu = mean_axis(t, x, axis)?;
        let mu = t.broadcast_axis(mu, axis, extent)?;
        t.binary(BinOp::Sub, x, mu)?
    } else {
        x
    };
    let sq = t.binary(BinOp::Mul, centered, centered)?;
    Ok((centered, mean_axis(t, sq, axis)?))
}

/// `x op broadcast(y)`, right-aligned; the IR has no implicit broadcast.
fn broadcast_bin(t: &mut GraphTape<'_>, op: BinOp, x: Val, y: Val) -> Result<Val> {
    let shape = t.shape_of(x);
    let y = t.broadcast_to(y, &shape)?;
    t.binary(op, x, y)
}

/// `1 / sqrt(v + eps)` with `eps` a **uniform**, not a literal.
fn inv_sqrt_eps(t: &mut GraphTape<'_>, v: Val, eps: fusor_ir::shape::SymId) -> Result<Val> {
    let dtype = t.dtype_of(v);
    let body = ScalarExpr::un(
        UnOp::InverseSqrt,
        ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::arg(0, dtype),
            ScalarExpr::uniform(eps, dtype),
        ),
    );
    t.map(body, &[v])
}

/// Optional mean-centre, `/sqrt(mean(x^2) + eps)`, `*w`, `+b` over the last
/// axis. Without the centring this is RMS normalization.
pub(crate) fn layer_norm_defn(
    t: &mut GraphTape<'_>,
    x: Val,
    weight: Option<Val>,
    bias: Option<Val>,
    eps: fusor_ir::shape::SymId,
    remove_mean: bool,
) -> Result<Val> {
    let rank = t.rank_of(x);
    let axis = rank
        .checked_sub(1)
        .ok_or_else(|| Error::Shape("normalization needs at least a rank-1 value".into()))?
        as u32;
    let extent = t.shape_of(x)[axis as usize];
    let (centered, var) = centered_variance(t, x, axis, remove_mean)?;
    let inv = inv_sqrt_eps(t, var, eps)?;
    let inv = t.broadcast_axis(inv, axis, extent)?;
    let mut y = t.binary(BinOp::Mul, centered, inv)?;
    if let Some(w) = weight {
        y = broadcast_bin(t, BinOp::Mul, y, w)?;
    }
    if let Some(b) = bias {
        y = broadcast_bin(t, BinOp::Add, y, b)?;
    }
    Ok(y)
}

/// A `SymId` for one epsilon value, shared between two layers that use the
/// same one. `eps` is a uniform so it stays out of the kernel's identity and
/// binding list.
pub(crate) fn eps_uniform(graph: &GraphRef, eps: f32) -> fusor_ir::shape::SymId {
    let sym = graph.named_sym(&format!("eps#{:08x}", eps.to_bits()));
    graph.set_uniform(sym, eps);
    sym
}

fn last_axis(graph: &GraphRef, x: &Tensor) -> Result<u32> {
    let rank = graph.facts(x.id).rank();
    rank.checked_sub(1)
        .map(|a| a as u32)
        .ok_or_else(|| Error::Shape("a rank-0 value has no last axis".into()))
}

impl Tensor {
    /// Softmax over `axis`.
    pub fn softmax(&self, axis: u32) -> Result<Tensor> {
        let x = self.id;
        core_op(&self.graph, |t| softmax_defn(t, x, axis))
    }

    /// Softmax over the last axis.
    pub fn softmax_last_dim(&self) -> Result<Tensor> {
        self.softmax(last_axis(&self.graph, self)?)
    }

    /// Log-softmax over `axis`.
    pub fn log_softmax(&self, axis: u32) -> Result<Tensor> {
        let x = self.id;
        core_op(&self.graph, |t| log_softmax_defn(t, x, axis))
    }

    /// `x / sqrt(mean(x^2) + eps) * weight`.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor> {
        self.rms_norm_inner(Some(weight), None, eps, None)
    }

    /// RMS normalization without a learned scale.
    pub fn rms_norm_no_weight(&self, eps: f32) -> Result<Tensor> {
        self.rms_norm_inner(None, None, eps, None)
    }

    /// [`Tensor::rms_norm`] with a learned shift as well as a scale.
    pub fn rms_norm_with_bias(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        self.rms_norm_inner(Some(weight), Some(bias), eps, None)
    }

    /// The transformer block boundary: `rms_norm(x + residual)`.
    pub fn rms_norm_residual(
        &self,
        residual: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        if self
            .graph
            .session()
            .caps()
            .execution_profile
            .enabled("FUSOR_PROFILE_RMS_RESIDUAL")
        {
            return self.rms_norm_inner(Some(weight), bias, eps, Some(residual));
        }
        let (x, r) = (self.id, residual.id);
        let sum = core_op(&self.graph, |t| {
            let shape = t.shape_of(x);
            let r = t.broadcast_to(r, &shape)?;
            t.binary(BinOp::Add, x, r)
        })?;
        sum.rms_norm_inner(Some(weight), bias, eps, None)
    }

    fn rms_norm_inner(
        &self,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let sym = eps_uniform(&self.graph, eps);
        let x = self.id;
        let w = weight.map(|t| t.id);
        let b = bias.map(|t| t.id);
        let r = residual.map(|t| t.id);
        let caps = self.graph.session().caps();
        let rms_block = self
            .graph
            .session()
            .caps()
            .execution_profile
            .var("FUSOR_PROFILE_RMS_BLOCK")
            .ok()
            .map(|v| v.parse::<u32>().expect("RMS block must be an integer"))
            .unwrap_or(256);
        assert!(
            matches!(rms_block, 32 | 64 | 128 | 256 | 512 | 1024),
            "RMS block must be a power of two from 32 to 1024"
        );
        core_op(&self.graph, |t| {
            let summed = if let Some(r) = r {
                let shape = t.shape_of(x);
                let r = t.broadcast_to(r, &shape)?;
                t.binary(BinOp::Add, x, r)?
            } else {
                x
            };
            let result = layer_norm_defn(t, summed, w, b, sym, false)?;
            let shape = t.shape_of(x);
            if self
                .graph
                .session()
                .caps()
                .execution_profile
                .enabled("FUSOR_PROFILE_FUSED_RMS_NORM")
                && caps.kind == fusor_ir::device::DeviceKind::Gpu
                && caps.limits.max_compute_invocations_per_workgroup >= rms_block
                && caps.limits.max_compute_workgroup_size[0] >= rms_block
                && caps.limits.max_compute_workgroup_storage_size >= rms_block * 4
                && b.is_none()
                && let Some(w) = w
                && let Some(width) = shape.last().and_then(|d| d.as_const())
                && (1..=8192).contains(&width)
                && shape
                    .iter()
                    .try_fold(1u64, |a, d| a.checked_mul(d.as_const()?))
                    .is_some_and(|n| n > 0 && n <= width * 4)
                && t.shape_of(w).as_slice() == [Dim::Const(width)]
                && r.is_none_or(|r| t.shape_of(r) == shape)
                && [x, w].into_iter().chain(r).all(|v| {
                    t.dtype_of(v) == fusor_ir::dtype::Dtype::F32
                        && t.facts(v).numeric.reassoc
                        && t.facts(v).numeric.contract
                })
            {
                use fusor_ir::ir::launch::{
                    AccessPlan, IndexSpace, Launch, Operand, ScheduleDomain,
                };
                use fusor_ir::shape::Layout;
                let ops = [x, w]
                    .into_iter()
                    .chain(r)
                    .map(|src| Operand {
                        src,
                        layout: Layout::contiguous(&t.shape_of(src)),
                        access: AccessPlan::Alias,
                    })
                    .collect();
                let fused = t.graph_mut().builder(&caps).add_launch(Launch::RmsNorm {
                    output: IndexSpace::new(shape),
                    width: width as u32,
                    block: rms_block,
                    epsilon: ScalarExpr::uniform(sym, fusor_ir::dtype::Dtype::F32),
                    ops,
                    sched: ScheduleDomain::Point,
                })?;
                // Keep the logical root so reverse-mode differentiation still
                // sees ordinary arithmetic; extraction can select this launch.
                t.graph_mut().builder(&caps).union(result, fused)?;
            }
            Ok(result)
        })
    }

    /// `(x - mean) / sqrt(var + eps) * weight + bias` over the last axis.
    /// `remove_mean == false` is the RMS-like spelling.
    pub fn layer_norm(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
        remove_mean: bool,
    ) -> Result<Tensor> {
        let sym = eps_uniform(&self.graph, eps);
        let x = self.id;
        let w = weight.id;
        let b = bias.map(|t| t.id);
        core_op(&self.graph, |t| {
            layer_norm_defn(t, x, Some(w), b, sym, remove_mean)
        })
    }

    /// `mean(x^2)`-free variance over the last axis, for callers that want the
    /// statistic rather than the normalized value.
    pub fn variance_last(&self) -> Result<Tensor> {
        let x = self.id;
        let axis = last_axis(&self.graph, self)?;
        core_op(&self.graph, |t| Ok(centered_variance(t, x, axis, true)?.1))
    }
}
