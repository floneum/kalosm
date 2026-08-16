//! Dtype conversion and rounding. `cast` is a `ScalarExpr` node inside a
//! `Map`, differentiable both directions by `map_adjoint`.
//!
//! Every pair among `{F32, F16, BF16, U32, I32}` is legal, and
//! `round`/`floor`/`ceil`/`trunc` are real primitives.

use fusor_ir::dtype::{Dtype, RoundMode};
use fusor_ir::scalar::{BinOp, ScalarExpr};

use crate::tensor::{Scalar, Tensor};
use crate::{Error, Result};

/// `fake_quant`'s adjoint: the identity into `x`, an explicit zero into every
/// other operand. See [`Tensor::fake_quant`].
fn straight_through_in_x(
    tape: &mut dyn fusor_ir::autograd::Tape,
    _node: &fusor_ir::ir::Node,
    grad: fusor_ir::autograd::Val,
    ins: &[fusor_ir::autograd::Val],
    _out: fusor_ir::autograd::Val,
) -> fusor_ir::Result<fusor_ir::autograd::Grads> {
    let mut grads: fusor_ir::autograd::Grads = smallvec::smallvec![Some(grad)];
    for v in ins.iter().skip(1) {
        let zero = tape.zeros_like(*v)?;
        grads.push(Some(zero));
    }
    Ok(grads)
}

impl Tensor {
    /// Numeric conversion. One `Map`.
    pub fn cast(&self, to: Dtype) -> Result<Tensor> {
        self.require_dense("cast")?;
        if to.is_quantized() {
            return Err(Error::Dtype(
                "cast to a quantized dtype is a quantize op, not a scalar cast".into(),
            ));
        }
        self.map1(ScalarExpr::cast(to, self.arg0()))
    }

    /// Convert elements to `f32`.
    pub fn to_f32(&self) -> Result<Tensor> {
        self.cast(Dtype::F32)
    }
    /// Convert elements to `u32`.
    pub fn to_u32(&self) -> Result<Tensor> {
        self.cast(Dtype::U32)
    }
    /// Convert elements to `i32`.
    pub fn to_i32(&self) -> Result<Tensor> {
        self.cast(Dtype::I32)
    }

    /// Reinterpret the bits. Never differentiable; widths must match.
    pub fn bitcast(&self, to: Dtype) -> Result<Tensor> {
        self.require_dense("bitcast")?;
        if to.byte_size() != self.dtype().byte_size() {
            return Err(Error::Dtype(format!(
                "bitcast {:?} -> {to:?} changes the element width",
                self.dtype()
            )));
        }
        self.map1(ScalarExpr::bitcast(to, self.arg0()))
    }

    /// Round with an explicit mode.
    ///
    /// **Known gap:** Logical has no carrier for a per-node `NumericContract`, so
    /// the "this value is `STRICT`, do not fast-math it" obligation currently
    /// rides on `ScalarKind::Round` itself and must be honoured by the
    /// emitter.
    pub fn round_mode(&self, mode: RoundMode) -> Result<Tensor> {
        self.require_dense("round")?;
        self.map1(ScalarExpr::round(mode, self.arg0()))
    }

    /// Round half away from zero. MSQ1 export idempotence depends on this
    /// mode; `tf.round` disagrees with it on an exact `.5`.
    pub fn round(&self) -> Result<Tensor> {
        self.round_mode(RoundMode::HalfAwayFromZero)
    }
    /// Round half to even.
    pub fn round_even(&self) -> Result<Tensor> {
        self.round_mode(RoundMode::HalfToEven)
    }
    /// Round toward negative infinity.
    pub fn floor(&self) -> Result<Tensor> {
        self.round_mode(RoundMode::Floor)
    }
    /// Round toward positive infinity.
    pub fn ceil(&self) -> Result<Tensor> {
        self.round_mode(RoundMode::Ceil)
    }
    /// Round toward zero.
    pub fn trunc(&self) -> Result<Tensor> {
        self.round_mode(RoundMode::Trunc)
    }

    /// QAT fake-quant forward: `round(x / scale).clamp(-levels, levels) *
    /// scale`, with `scale` broadcast in.
    ///
    /// One `Map` with a straight-through backward rule: the incoming gradient
    /// routes to operand 0 unchanged, which requires operand 0 to *be* `x`.
    /// The scale receives an explicit zero, not nothing — the walk treats
    /// every float leaf as trainable, so omitting it would starve the scale's
    /// subgraph.
    pub fn fake_quant(&self, levels: u32, scale: &Tensor) -> Result<Tensor> {
        if levels == 0 {
            return Err(Error::Shape("fake_quant needs levels > 0".into()));
        }
        let lim = levels as f32;
        let dt = self.dtype();
        let (x, s, _) = crate::broadcast::broadcast_pair(self, scale)?;
        let q = ScalarExpr::round(
            RoundMode::HalfAwayFromZero,
            ScalarExpr::bin(BinOp::Div, ScalarExpr::arg(0, dt), ScalarExpr::arg(1, dt)),
        );
        let clamped = ScalarExpr::bin(
            BinOp::Min,
            ScalarExpr::bin(BinOp::Max, q, Scalar::from(-lim).expr(dt)),
            Scalar::from(lim).expr(dt),
        );
        let body = ScalarExpr::bin(BinOp::Mul, clamped, ScalarExpr::arg(1, dt));
        let y = Tensor::mapn(&self.graph, body, &[&x, &s])?;
        let parents = [
            crate::graph::parent(&x, true),
            crate::graph::parent(&s, true),
        ];
        self.graph
            .register_backward(y.id, &parents, straight_through_in_x)?;
        Ok(y)
    }
}
