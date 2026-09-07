//! `relu`, `sigmoid`, `silu`, `gelu`, `tanh_exact`, `softplus`. Each is one
//! `Map` with a different `ScalarExpr`; none is a kernel.

use fusor_autograd::tape::splat_of;
use fusor_ir::dtype::Dtype;
use fusor_ir::scalar::{BinOp, CmpOp, ScalarExpr, UnOp};
use fusor_ir::{Error, Result};

use crate::tensor::Tensor;

/// `sqrt(2/pi)`, the tanh-GELU coefficient.
const GELU_C: f32 = 0.797_884_6;
/// The cubic term's coefficient.
const GELU_K: f32 = 0.044_715;
/// Clamp value to avoid numerical issues with `tanh`.
const TANH_CLAMP: f32 = 15.0;

fn lit(dtype: Dtype, v: f32) -> Result<ScalarExpr> {
    Ok(ScalarExpr::lit(splat_of(dtype, v)?))
}

fn mul(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::bin(BinOp::Mul, a, b)
}
fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::bin(BinOp::Add, a, b)
}
fn sub(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::bin(BinOp::Sub, a, b)
}
fn div(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::bin(BinOp::Div, a, b)
}

/// `min(max(x, lo), hi)`.
fn clamp(x: ScalarExpr, lo: ScalarExpr, hi: ScalarExpr) -> ScalarExpr {
    ScalarExpr::bin(BinOp::Min, ScalarExpr::bin(BinOp::Max, x, lo), hi)
}

// The expression builders are public so a rewrite rule can recognize the
// exact tree the frontend emits, and conformance can compare two spellings.

/// `max(x, 0)`.
fn relu_expr(dtype: Dtype) -> Result<ScalarExpr> {
    Ok(ScalarExpr::bin(
        BinOp::Max,
        ScalarExpr::arg(0, dtype),
        lit(dtype, 0.0)?,
    ))
}

/// `1 / (1 + exp(-x))`.
fn sigmoid_expr(dtype: Dtype) -> Result<ScalarExpr> {
    let x = ScalarExpr::arg(0, dtype);
    let e = ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, x));
    Ok(div(lit(dtype, 1.0)?, add(lit(dtype, 1.0)?, e)))
}

/// `x / (1 + exp(-x))`.
fn silu_expr(dtype: Dtype) -> Result<ScalarExpr> {
    let x = ScalarExpr::arg(0, dtype);
    let e = ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, x.clone()));
    Ok(div(x, add(lit(dtype, 1.0)?, e)))
}

/// `(e^x - e^-x) / (e^x + e^-x)`, written out instead of the driver's `tanh`:
/// WARP under-saturates the negative tail of the native `tanh` and the GELU
/// tail depends on it.
fn tanh_exact_expr(dtype: Dtype) -> Result<ScalarExpr> {
    Ok(tanh_exact_of(ScalarExpr::arg(0, dtype)))
}

fn tanh_exact_of(x: ScalarExpr) -> ScalarExpr {
    let p = ScalarExpr::un(UnOp::Exp, x.clone());
    let n = ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, x));
    div(sub(p.clone(), n.clone()), add(p, n))
}

/// The tanh approximation, ported verbatim from the reference including its
/// three defensive clamps:
///
/// `0.5*x*(1 + clamp(tanh_exact(clamp(c*(x + k*x^3), -15, 15)), -1, 1))`,
/// with `1 + tanh` clamped to `[0, 2]`.
///
/// The clamps are what keep the value finite at +/-20 on every driver.
fn gelu_expr(dtype: Dtype) -> Result<ScalarExpr> {
    let x = ScalarExpr::arg(0, dtype);
    let x3 = mul(x.clone(), mul(x.clone(), x.clone()));
    let inner = mul(
        lit(dtype, GELU_C)?,
        add(x.clone(), mul(lit(dtype, GELU_K)?, x3)),
    );
    let inner = clamp(inner, lit(dtype, -TANH_CLAMP)?, lit(dtype, TANH_CLAMP)?);
    let t = clamp(tanh_exact_of(inner), lit(dtype, -1.0)?, lit(dtype, 1.0)?);
    let one_plus = clamp(add(lit(dtype, 1.0)?, t), lit(dtype, 0.0)?, lit(dtype, 2.0)?);
    Ok(mul(mul(lit(dtype, 0.5)?, x), one_plus))
}

/// `0.5 * x * (1 + erf(x / sqrt 2))` with erf from Abramowitz & Stegun 7.1.26
/// (max absolute error 1.5e-7).
fn gelu_exact_expr(dtype: Dtype) -> Result<ScalarExpr> {
    const P: f32 = 0.327_591_1;
    const A: [f32; 5] = [
        0.254_829_6,
        -0.284_496_72,
        1.421_413_8,
        -1.453_152_1,
        1.061_405_4,
    ];
    let x = ScalarExpr::arg(0, dtype);
    let z = mul(x.clone(), lit(dtype, std::f32::consts::FRAC_1_SQRT_2)?);
    let a = ScalarExpr::un(UnOp::Abs, z.clone());
    let t = div(
        lit(dtype, 1.0)?,
        add(lit(dtype, 1.0)?, mul(lit(dtype, P)?, a.clone())),
    );
    // Horner over t, times exp(-a^2).
    let mut poly = lit(dtype, A[4])?;
    for c in A[..4].iter().rev() {
        poly = add(lit(dtype, *c)?, mul(poly, t.clone()));
    }
    let poly = mul(poly, t);
    let decay = ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, mul(a.clone(), a)));
    let magnitude = sub(lit(dtype, 1.0)?, mul(poly, decay));
    // erf is odd; recover the sign of z without a `sign` opcode.
    let negative = ScalarExpr::cmp(CmpOp::Lt, z, lit(dtype, 0.0)?);
    let erf = ScalarExpr::select(
        negative,
        ScalarExpr::un(UnOp::Neg, magnitude.clone()),
        magnitude,
    );
    Ok(mul(mul(lit(dtype, 0.5)?, x), add(lit(dtype, 1.0)?, erf)))
}

/// `max(x, 0) + log(1 + exp(-|x|))` — the numerically stable softplus. Its
/// derivative is a single sigmoid, which is what the distillation loss's
/// adjoint rewrite collapses the taped chain into.
fn softplus_expr(dtype: Dtype) -> Result<ScalarExpr> {
    let x = ScalarExpr::arg(0, dtype);
    let a = ScalarExpr::un(UnOp::Abs, x.clone());
    let tail = ScalarExpr::un(
        UnOp::Log,
        add(
            lit(dtype, 1.0)?,
            ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, a)),
        ),
    );
    Ok(add(ScalarExpr::bin(BinOp::Max, x, lit(dtype, 0.0)?), tail))
}

/// `x > 0 ? x : slope * x`.
fn leaky_relu_expr(dtype: Dtype, slope: f32) -> Result<ScalarExpr> {
    let x = ScalarExpr::arg(0, dtype);
    let positive = ScalarExpr::cmp(CmpOp::Gt, x.clone(), lit(dtype, 0.0)?);
    Ok(ScalarExpr::select(
        positive,
        x.clone(),
        mul(lit(dtype, slope)?, x),
    ))
}

impl Tensor {
    fn activation(&self, build: impl FnOnce(Dtype) -> Result<ScalarExpr>) -> Result<Tensor> {
        let dtype = self.graph.facts(self.id).dtype;
        if !dtype.is_float() {
            return Err(Error::Dtype(format!(
                "an activation needs a float operand, got {dtype:?}"
            )));
        }
        let expr = build(dtype)?;
        let id = self
            .graph
            .build(|t| fusor_ir::autograd::Tape::map(t, expr, &[self.id]))?;
        Ok(self.graph.tensor(id))
    }

    /// Rectified linear activation.
    pub fn relu(&self) -> Result<Tensor> {
        self.activation(relu_expr)
    }

    /// Logistic sigmoid activation.
    pub fn sigmoid(&self) -> Result<Tensor> {
        self.activation(sigmoid_expr)
    }

    /// Sigmoid linear unit activation.
    pub fn silu(&self) -> Result<Tensor> {
        self.activation(silu_expr)
    }

    /// The tanh approximation, matching the reference's default.
    pub fn gelu(&self) -> Result<Tensor> {
        self.activation(gelu_expr)
    }

    /// The erf-exact form.
    pub fn gelu_exact(&self) -> Result<Tensor> {
        self.activation(gelu_exact_expr)
    }

    /// Hyperbolic tangent activation.
    pub fn tanh_exact(&self) -> Result<Tensor> {
        self.activation(tanh_exact_expr)
    }

    /// Numerically stable softplus activation.
    pub fn softplus(&self) -> Result<Tensor> {
        self.activation(softplus_expr)
    }

    /// Leaky rectified linear activation with the given negative slope.
    pub fn leaky_relu(&self, slope: f32) -> Result<Tensor> {
        self.activation(|d| leaky_relu_expr(d, slope))
    }
}
