//! `relu`, `sigmoid`, `silu`, `gelu`, `tanh_exact`, `softplus`. Each is one
//! `Map` with a different `ScalarExpr`; none is a kernel.

use fusor2_autograd::tape::splat_of;
use fusor2_ir::dtype::Dtype;
use fusor2_ir::scalar::{BinOp, CmpOp, ScalarExpr, UnOp};
use fusor2_ir::{Error, Result};

use crate::tensor::Tensor;

/// `sqrt(2/pi)`, the tanh-GELU coefficient.
const GELU_C: f32 = 0.797_884_56;
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
pub fn relu_expr(dtype: Dtype) -> Result<ScalarExpr> {
    Ok(ScalarExpr::bin(
        BinOp::Max,
        ScalarExpr::arg(0, dtype),
        lit(dtype, 0.0)?,
    ))
}

/// `1 / (1 + exp(-x))`.
pub fn sigmoid_expr(dtype: Dtype) -> Result<ScalarExpr> {
    let x = ScalarExpr::arg(0, dtype);
    let e = ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, x));
    Ok(div(lit(dtype, 1.0)?, add(lit(dtype, 1.0)?, e)))
}

/// `x / (1 + exp(-x))`.
pub fn silu_expr(dtype: Dtype) -> Result<ScalarExpr> {
    let x = ScalarExpr::arg(0, dtype);
    let e = ScalarExpr::un(UnOp::Exp, ScalarExpr::un(UnOp::Neg, x.clone()));
    Ok(div(x, add(lit(dtype, 1.0)?, e)))
}

/// `(e^x - e^-x) / (e^x + e^-x)`, written out instead of the driver's `tanh`:
/// WARP under-saturates the negative tail of the native `tanh` and the GELU
/// tail depends on it.
pub fn tanh_exact_expr(dtype: Dtype) -> Result<ScalarExpr> {
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
pub fn gelu_expr(dtype: Dtype) -> Result<ScalarExpr> {
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
pub fn gelu_exact_expr(dtype: Dtype) -> Result<ScalarExpr> {
    const P: f32 = 0.327_591_1;
    const A: [f32; 5] = [
        0.254_829_592,
        -0.284_496_736,
        1.421_413_741,
        -1.453_152_027,
        1.061_405_429,
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
pub fn softplus_expr(dtype: Dtype) -> Result<ScalarExpr> {
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
pub fn leaky_relu_expr(dtype: Dtype, slope: f32) -> Result<ScalarExpr> {
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
            .build(|t| fusor2_ir::autograd::Tape::map(t, expr, &[self.id]))?;
        Ok(self.graph.tensor(id))
    }

    pub fn relu(&self) -> Result<Tensor> {
        self.activation(relu_expr)
    }

    pub fn sigmoid(&self) -> Result<Tensor> {
        self.activation(sigmoid_expr)
    }

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

    pub fn tanh_exact(&self) -> Result<Tensor> {
        self.activation(tanh_exact_expr)
    }

    pub fn softplus(&self) -> Result<Tensor> {
        self.activation(softplus_expr)
    }

    pub fn leaky_relu(&self, slope: f32) -> Result<Tensor> {
        self.activation(|d| leaky_relu_expr(d, slope))
    }
}

/// Evaluate a single-argument `ScalarExpr` on the host.
///
/// The reference a backend is compared against — a plain tree walk, never a
/// path a kernel takes. Public because `fusor2-conformance` uses the same
/// walk.
pub fn eval_host(expr: &ScalarExpr, arg: f32) -> f32 {
    use fusor2_ir::dtype::Splat;
    use fusor2_ir::scalar::ScalarKind;
    match expr.kind() {
        ScalarKind::Arg(_) => arg,
        ScalarKind::Lit(l) => match l.0 {
            Splat::F32(v) => v,
            Splat::F16(v) => half::f16::from_bits(v).to_f32(),
            Splat::BF16(v) => half::bf16::from_bits(v).to_f32(),
            Splat::U32(v) => v as f32,
            Splat::I32(v) => v as f32,
        },
        ScalarKind::Un { op, x } => {
            let v = eval_host(x, arg);
            match op {
                UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp => v.exp(),
                UnOp::Exp2 => v.exp2(),
                UnOp::Log => v.ln(),
                UnOp::Log2 => v.log2(),
                UnOp::Sqrt => v.sqrt(),
                UnOp::InverseSqrt => 1.0 / v.sqrt(),
                UnOp::Sin => v.sin(),
                UnOp::Cos => v.cos(),
                UnOp::Tan => v.tan(),
                UnOp::Tanh => v.tanh(),
                UnOp::Asin => v.asin(),
                UnOp::Acos => v.acos(),
                UnOp::Atan => v.atan(),
                UnOp::Sinh => v.sinh(),
                UnOp::Cosh => v.cosh(),
                UnOp::Asinh => v.asinh(),
                UnOp::Acosh => v.acosh(),
                UnOp::Atanh => v.atanh(),
                UnOp::Abs => v.abs(),
                UnOp::Neg => -v,
                UnOp::Unpack2x16Float => v,
            }
        }
        ScalarKind::Bin { op, a, b } => {
            let (x, y) = (eval_host(a, arg), eval_host(b, arg));
            match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
                BinOp::Rem => x % y,
                BinOp::Pow => x.powf(y),
                BinOp::Min => x.min(y),
                BinOp::Max => x.max(y),
                _ => f32::NAN,
            }
        }
        ScalarKind::Cmp { op, a, b } => {
            let (x, y) = (eval_host(a, arg), eval_host(b, arg));
            let r = match op {
                CmpOp::Lt => x < y,
                CmpOp::Le => x <= y,
                CmpOp::Gt => x > y,
                CmpOp::Ge => x >= y,
                CmpOp::Eq => x == y,
                CmpOp::Ne => x != y,
            };
            if r { 1.0 } else { 0.0 }
        }
        ScalarKind::Select { c, t, f } => {
            if eval_host(c, arg) != 0.0 {
                eval_host(t, arg)
            } else {
                eval_host(f, arg)
            }
        }
        ScalarKind::Cast { x, .. } | ScalarKind::Bitcast { x, .. } => eval_host(x, arg),
        ScalarKind::Round { mode, x } => {
            let v = eval_host(x, arg);
            match mode {
                fusor2_ir::dtype::RoundMode::Floor => v.floor(),
                fusor2_ir::dtype::RoundMode::Ceil => v.ceil(),
                fusor2_ir::dtype::RoundMode::Trunc => v.trunc(),
                fusor2_ir::dtype::RoundMode::HalfAwayFromZero => v.round(),
                fusor2_ir::dtype::RoundMode::HalfToEven => {
                    let r = v.round();
                    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
                        r - v.signum()
                    } else {
                        r
                    }
                }
            }
        }
        ScalarKind::Uniform(_) | ScalarKind::IndexOf(_) => f32::NAN,
        ScalarKind::Dot { .. } | ScalarKind::Splat { .. } => f32::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_gelu(x: f32) -> f32 {
        let inner = (GELU_C * (x + GELU_K * x * x * x)).clamp(-TANH_CLAMP, TANH_CLAMP);
        let t = inner.tanh().clamp(-1.0, 1.0);
        0.5 * x * (1.0 + t).clamp(0.0, 2.0)
    }

    #[test]
    fn gelu_matches_the_reference_formula_and_is_finite_at_the_tails() {
        let e = gelu_expr(Dtype::F32).unwrap();
        for x in [-20.0f32, -1.0, 0.0, 1.0, 20.0] {
            let got = eval_host(&e, x);
            let want = reference_gelu(x);
            assert!(got.is_finite(), "gelu({x}) = {got}");
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "gelu({x}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn tanh_exact_agrees_with_the_host_tanh_away_from_saturation() {
        let e = tanh_exact_expr(Dtype::F32).unwrap();
        for x in [-3.0f32, -0.5, 0.0, 0.5, 3.0] {
            assert!((eval_host(&e, x) - x.tanh()).abs() < 1e-6);
        }
    }

    #[test]
    fn sigmoid_silu_relu_and_softplus_are_the_expected_scalars() {
        let s = sigmoid_expr(Dtype::F32).unwrap();
        let u = silu_expr(Dtype::F32).unwrap();
        let r = relu_expr(Dtype::F32).unwrap();
        let p = softplus_expr(Dtype::F32).unwrap();
        for x in [-4.0f32, -1.0, 0.0, 0.7, 5.0] {
            let sig = 1.0 / (1.0 + (-x).exp());
            assert!((eval_host(&s, x) - sig).abs() < 1e-6);
            assert!((eval_host(&u, x) - x * sig).abs() < 1e-6);
            assert!((eval_host(&r, x) - x.max(0.0)).abs() < 1e-7);
            assert!((eval_host(&p, x) - (1.0 + x.exp()).ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn exact_gelu_tracks_the_erf_form() {
        let e = gelu_exact_expr(Dtype::F32).unwrap();
        for x in [-3.0f32, -1.0, 0.0, 1.0, 3.0] {
            let z = x * std::f32::consts::FRAC_1_SQRT_2;
            let a = z.abs();
            let t = 1.0 / (1.0 + 0.327_591_1 * a);
            let poly = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t
                - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t;
            let mag = 1.0 - poly * (-a * a).exp();
            let erf = if z < 0.0 { -mag } else { mag };
            let want = 0.5 * x * (1.0 + erf);
            assert!((eval_host(&e, x) - want).abs() < 1e-5, "x = {x}");
        }
    }

    #[test]
    fn leaky_relu_passes_positives_and_scales_negatives() {
        let e = leaky_relu_expr(Dtype::F32, 0.01).unwrap();
        assert!((eval_host(&e, 2.0) - 2.0).abs() < 1e-7);
        assert!((eval_host(&e, -2.0) + 0.02).abs() < 1e-7);
    }
}
