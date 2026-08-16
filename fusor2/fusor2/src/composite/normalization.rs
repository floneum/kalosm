//! softmax, rms_norm and layer_norm. All macro ops over `Fold` + `Map`, fused
//! into one launch by `fold_split` + `map_into_fold`.
//!
//! Every softmax spelling shares **one** `defn`, under a sugar node minted in
//! the same call.

use fusor2_autograd::tape::{GraphTape, TapeExt, accum_dtype};
use fusor2_ir::autograd::{Tape, Val};
use fusor2_ir::scalar::{BinOp, ScalarExpr, UnOp};
use fusor2_ir::{Error, Result};

use crate::composite::{MacroAttr, MacroOp, NormKind, core_op, macro_op};
use crate::graph::GraphRef;
use crate::tensor::Tensor;

/// `exp(x - max) / sum(exp(x - max))` over `axis`. Every entry point shares
/// this.
pub(crate) fn softmax_defn(t: &mut GraphTape<'_>, x: Val, axis: u32) -> Result<Val> {
    let shape = t.shape_of(x);
    let dtype = t.dtype_of(x);
    let extent = *shape
        .get(axis as usize)
        .ok_or_else(|| Error::Shape(format!("softmax axis {axis} out of range")))?;

    let m = t.fold_binop(BinOp::Max, axis, dtype, x)?;
    let m = t.broadcast_axis(m, axis, extent)?;
    let centered = t.binary(BinOp::Sub, x, m)?;
    let e = t.unary(UnOp::Exp, centered)?;

    let acc = accum_dtype(dtype);
    let s = t.fold_binop(BinOp::Add, axis, acc, e)?;
    let s = t.cast(dtype, s)?;
    let s = t.broadcast_axis(s, axis, extent)?;
    t.binary(BinOp::Div, e, s)
}

/// `x - max - log(sum(exp(x - max)))`.
pub(crate) fn log_softmax_defn(t: &mut GraphTape<'_>, x: Val, axis: u32) -> Result<Val> {
    let shape = t.shape_of(x);
    let dtype = t.dtype_of(x);
    let extent = *shape
        .get(axis as usize)
        .ok_or_else(|| Error::Shape(format!("log_softmax axis {axis} out of range")))?;

    let m = t.fold_binop(BinOp::Max, axis, dtype, x)?;
    let mb = t.broadcast_axis(m, axis, extent)?;
    let centered = t.binary(BinOp::Sub, x, mb)?;
    let e = t.unary(UnOp::Exp, centered.clone())?;
    let acc = accum_dtype(dtype);
    let s = t.fold_binop(BinOp::Add, axis, acc, e)?;
    let s = t.cast(dtype, s)?;
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

/// `x op broadcast(y)`, right-aligned; the IR has no implicit broadcast.
fn broadcast_bin(t: &mut GraphTape<'_>, op: BinOp, x: Val, y: Val) -> Result<Val> {
    let shape = t.shape_of(x);
    let y = t.broadcast_to(y, &shape)?;
    t.binary(op, x, y)
}

/// `1 / sqrt(v + eps)` with `eps` a **uniform**, not a literal.
fn inv_sqrt_eps(t: &mut GraphTape<'_>, v: Val, eps: fusor2_ir::shape::SymId) -> Result<Val> {
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

/// `x / sqrt(mean(x^2) + eps) * w [+ b]` over the last axis.
pub(crate) fn rms_norm_defn(
    t: &mut GraphTape<'_>,
    x: Val,
    weight: Option<Val>,
    bias: Option<Val>,
    eps: fusor2_ir::shape::SymId,
) -> Result<Val> {
    let rank = t.rank_of(x);
    let axis = rank.checked_sub(1).ok_or_else(|| {
        Error::Shape("rms_norm needs at least a rank-1 value".into())
    })? as u32;
    let sq = t.binary(BinOp::Mul, x, x)?;
    let ms = mean_axis(t, sq, axis)?;
    let inv = inv_sqrt_eps(t, ms, eps)?;
    let extent = t.shape_of(x)[axis as usize];
    let inv = t.broadcast_axis(inv, axis, extent)?;
    let mut y = t.binary(BinOp::Mul, x, inv)?;
    if let Some(w) = weight {
        y = broadcast_bin(t, BinOp::Mul, y, w)?;
    }
    if let Some(b) = bias {
        y = broadcast_bin(t, BinOp::Add, y, b)?;
    }
    Ok(y)
}

/// Optional mean-centre, biased variance, `/sqrt(var + eps)`, `*w`, `+b`.
pub(crate) fn layer_norm_defn(
    t: &mut GraphTape<'_>,
    x: Val,
    weight: Option<Val>,
    bias: Option<Val>,
    eps: fusor2_ir::shape::SymId,
    remove_mean: bool,
) -> Result<Val> {
    let rank = t.rank_of(x);
    let axis = rank.checked_sub(1).ok_or_else(|| {
        Error::Shape("layer_norm needs at least a rank-1 value".into())
    })? as u32;
    let extent = t.shape_of(x)[axis as usize];

    let centered = if remove_mean {
        let mu = mean_axis(t, x, axis)?;
        let mu = t.broadcast_axis(mu, axis, extent)?;
        t.binary(BinOp::Sub, x, mu)?
    } else {
        x
    };
    let sq = t.binary(BinOp::Mul, centered, centered)?;
    let var = mean_axis(t, sq, axis)?;
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
pub(crate) fn eps_uniform(graph: &GraphRef, eps: f32) -> fusor2_ir::shape::SymId {
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
    /// Softmax over `axis`, as a macro op: the sugar node carries the axis so
    /// a rule can read it, and the expansion is in the same class.
    pub fn softmax(&self, axis: u32) -> Result<Tensor> {
        let x = self.id;
        macro_op(
            &self.graph,
            MacroOp::Softmax,
            MacroAttr::Softmax { axis },
            &[x],
            |t| softmax_defn(t, x, axis),
        )
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
        self.rms_norm_inner(Some(weight), None, eps)
    }

    /// RMS normalization without a learned scale.
    pub fn rms_norm_no_weight(&self, eps: f32) -> Result<Tensor> {
        self.rms_norm_inner(None, None, eps)
    }

    /// [`Tensor::rms_norm`] with a learned shift as well as a scale.
    pub fn rms_norm_with_bias(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        self.rms_norm_inner(Some(weight), Some(bias), eps)
    }

    /// The transformer block boundary: `rms_norm(x + residual)`.
    pub fn rms_norm_residual(
        &self,
        residual: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let sym = eps_uniform(&self.graph, eps);
        let (x, r, w) = (self.id, residual.id, weight.id);
        let b = bias.map(|t| t.id);
        let mut ops = vec![x, r, w];
        ops.extend(b);
        macro_op(
            &self.graph,
            MacroOp::Norm,
            MacroAttr::Norm {
                kind: NormKind::Rms,
                eps: sym,
                remove_mean: false,
            },
            &ops,
            |t| {
                let shape = t.shape_of(x);
                let r = t.broadcast_to(r, &shape)?;
                let sum = t.binary(BinOp::Add, x, r)?;
                rms_norm_defn(t, sum, Some(w), b, sym)
            },
        )
    }

    fn rms_norm_inner(
        &self,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        let sym = eps_uniform(&self.graph, eps);
        let x = self.id;
        let w = weight.map(|t| t.id);
        let b = bias.map(|t| t.id);
        let mut ops = vec![x];
        ops.extend(w);
        ops.extend(b);
        macro_op(
            &self.graph,
            MacroOp::Norm,
            MacroAttr::Norm {
                kind: NormKind::Rms,
                eps: sym,
                remove_mean: false,
            },
            &ops,
            |t| rms_norm_defn(t, x, w, b, sym),
        )
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
        let mut ops = vec![x, w];
        ops.extend(b);
        macro_op(
            &self.graph,
            MacroOp::Norm,
            MacroAttr::Norm {
                kind: NormKind::Layer,
                eps: sym,
                remove_mean,
            },
            &ops,
            |t| layer_norm_defn(t, x, Some(w), b, sym, remove_mean),
        )
    }

    /// `mean(x^2)`-free variance over the last axis, for callers that want the
    /// statistic rather than the normalized value.
    pub fn variance_last(&self) -> Result<Tensor> {
        let x = self.id;
        let axis = last_axis(&self.graph, self)?;
        core_op(&self.graph, |t| {
            let extent = t.shape_of(x)[axis as usize];
            let mu = mean_axis(t, x, axis)?;
            let mu = t.broadcast_axis(mu, axis, extent)?;
            let c = t.binary(BinOp::Sub, x, mu)?;
            let sq = t.binary(BinOp::Mul, c, c)?;
            mean_axis(t, sq, axis)
        })
    }
}
