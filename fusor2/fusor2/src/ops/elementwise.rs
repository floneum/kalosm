//! The 23 elementwise unaries and the 6 same-rank / 5 broadcasting
//! binaries. Every one of these is **one `L0::Map` with a different
//! `ScalarExpr`** — there is no 50-variant opcode enum whose discriminant
//! ordering ends up load-bearing in a kernel cache key.
//!
//! Owned by W12.

use fusor2_ir::scalar::{BinOp, ScalarExpr, UnOp};

use crate::tensor::Tensor;
use crate::{Error, Result};

/// One method per row; each is `self.map1(Un(op, Arg0))`.
macro_rules! unary {
    ($($name:ident => $op:ident, $doc:expr;)*) => {
        impl Tensor {$(
            #[doc = $doc]
            pub fn $name(&self) -> Result<Tensor> {
                self.map1(ScalarExpr::un(UnOp::$op, self.arg0()))
            }
        )*}
    };
}

unary! {
    exp     => Exp,         "`e^x`. One `Map`.";
    exp2    => Exp2,        "`2^x`. One `Map`.";
    log     => Log,         "Natural log. One `Map`.";
    log2    => Log2,        "Base-2 log. One `Map`.";
    sqrt    => Sqrt,        "Square root. One `Map`.";
    inverse_sqrt => InverseSqrt, "`1/sqrt(x)`. One `Map`.";
    sin     => Sin,         "One `Map`.";
    cos     => Cos,         "One `Map`.";
    tan     => Tan,         "One `Map`.";
    tanh    => Tanh,        "Hyperbolic tangent. One `Map`.";
    asin    => Asin,        "One `Map`.";
    acos    => Acos,        "One `Map`.";
    atan    => Atan,        "One `Map`.";
    sinh    => Sinh,        "One `Map`.";
    cosh    => Cosh,        "One `Map`.";
    asinh   => Asinh,       "One `Map`.";
    acosh   => Acosh,       "One `Map`.";
    atanh   => Atanh,       "One `Map`.";
    abs     => Abs,         "Elementwise magnitude. One `Map`.";
    neg     => Neg,         "Elementwise negation. One `Map`.";
}

impl Tensor {
    /// `x * x`, as `Bin(Mul, Arg0, Arg0)` — still exactly one `Map`, and the
    /// operand is read once.
    pub fn sqr(&self) -> Result<Tensor> {
        let a = self.arg0();
        self.map1(ScalarExpr::bin(BinOp::Mul, a.clone(), a))
    }

    /// Alias of [`Tensor::sqr`].
    pub fn square(&self) -> Result<Tensor> {
        self.sqr()
    }

    /// `1 / x`.
    pub fn recip(&self) -> Result<Tensor> {
        let one = ScalarExpr::lit(crate::tensor::splat_one(self.dtype()));
        self.map1(ScalarExpr::bin(BinOp::Div, one, self.arg0()))
    }

    /// `sign(x)`: `(x > 0) - (x < 0)`, one `Map`.
    pub fn sign(&self) -> Result<Tensor> {
        use fusor2_ir::scalar::CmpOp;
        let zero = ScalarExpr::lit(crate::tensor::splat_zero(self.dtype()));
        let pos = ScalarExpr::cmp(CmpOp::Gt, self.arg0(), zero.clone());
        let neg = ScalarExpr::cmp(CmpOp::Lt, self.arg0(), zero);
        self.map1(ScalarExpr::bin(BinOp::Sub, pos, neg))
    }

    // NOTE: `tanh_exact` lives in W13's `composite/activations.rs`, which
    // builds the exact `(e^x - e^-x) / (e^x + e^-x)` form. Defining it here
    // as `Un(Tanh)` under a STRICT contract would collide, and L0 has no
    // carrier for a per-node `NumericContract` anyway — see the crate report.

    /// `exp` under a relaxed accuracy contract.
    ///
    /// Its **own** [`UnOp`], not sugar for [`Tensor::exp`]: hash-consing would
    /// otherwise merge a relaxed exponential with a strict one and a target
    /// could never substitute a cheaper sequence for the first without
    /// changing the second. Both currently lower to the target's exponential,
    /// which is what the reference does with `NaryOp::ApproximateExp`.
    pub fn approximate_exp(&self) -> Result<Tensor> {
        self.map1(ScalarExpr::un(UnOp::ApproximateExp, self.arg0()))
    }

    /// Medium-accuracy `exp`. See [`Tensor::approximate_exp`].
    pub fn less_approximate_exp(&self) -> Result<Tensor> {
        self.map1(ScalarExpr::un(UnOp::LessApproximateExp, self.arg0()))
    }

    // -- same-rank binaries ---------------------------------------------------

    fn bin_same(&self, rhs: &Tensor, op: BinOp, what: &str) -> Result<Tensor> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::Dtype(format!(
                "{what} operands differ in dtype: {:?} vs {:?}",
                self.dtype(),
                rhs.dtype()
            )));
        }
        let expr = ScalarExpr::bin(op, self.arg(0), rhs.arg(1));
        Tensor::mapn(&self.graph, expr, &[self, rhs])
    }

    /// Elementwise `a + b`; shapes must already match.
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_same(rhs, BinOp::Add, "add")
    }
    /// Elementwise `a - b`; shapes must already match.
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_same(rhs, BinOp::Sub, "sub")
    }
    /// Elementwise `a * b`; shapes must already match.
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_same(rhs, BinOp::Mul, "mul")
    }
    /// Elementwise `a / b`; shapes must already match.
    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_same(rhs, BinOp::Div, "div")
    }
    /// Elementwise `a % b`. Integer only, matching the reference's SIMD
    /// coverage.
    pub fn rem(&self, rhs: &Tensor) -> Result<Tensor> {
        if !self.dtype().is_int() {
            return Err(Error::Dtype(format!(
                "rem is defined on integer dtypes only, not {:?}",
                self.dtype()
            )));
        }
        self.bin_same(rhs, BinOp::Rem, "rem")
    }
    /// Elementwise `a ^ b`; shapes must already match.
    pub fn pow(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_same(rhs, BinOp::Pow, "pow")
    }
    /// Elementwise `max(a, b)`; shapes must already match.
    pub fn maximum(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_same(rhs, BinOp::Max, "maximum")
    }
    /// Elementwise `min(a, b)`; shapes must already match.
    pub fn minimum(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_same(rhs, BinOp::Min, "minimum")
    }

    // -- broadcasting binaries -------------------------------------------------

    fn bin_broadcast(&self, rhs: &Tensor, op: BinOp, what: &str) -> Result<Tensor> {
        let (a, b, _) = crate::broadcast::broadcast_pair(self, rhs)?;
        a.bin_same(&b, op, what)
    }

    /// Broadcasting `a + b`. Output rank is `max(rank_a, rank_b)`.
    pub fn add_(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_broadcast(rhs, BinOp::Add, "add_")
    }
    /// Broadcasting `a - b`.
    pub fn sub_(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_broadcast(rhs, BinOp::Sub, "sub_")
    }
    /// Broadcasting `a * b`.
    pub fn mul_(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_broadcast(rhs, BinOp::Mul, "mul_")
    }
    /// Broadcasting `a / b`.
    pub fn div_(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_broadcast(rhs, BinOp::Div, "div_")
    }
    /// Broadcasting `a ^ b`.
    pub fn pow_(&self, rhs: &Tensor) -> Result<Tensor> {
        self.bin_broadcast(rhs, BinOp::Pow, "pow_")
    }

    /// Scaffold spelling of [`Tensor::add_`].
    pub fn broadcast_add(&self, rhs: &Tensor) -> Result<Tensor> {
        self.add_(rhs)
    }
    /// Scaffold spelling of [`Tensor::sub_`].
    pub fn broadcast_sub(&self, rhs: &Tensor) -> Result<Tensor> {
        self.sub_(rhs)
    }
    /// Scaffold spelling of [`Tensor::mul_`].
    pub fn broadcast_mul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.mul_(rhs)
    }
    /// Scaffold spelling of [`Tensor::div_`].
    pub fn broadcast_div(&self, rhs: &Tensor) -> Result<Tensor> {
        self.div_(rhs)
    }
    /// Scaffold spelling of [`Tensor::pow_`].
    pub fn broadcast_pow(&self, rhs: &Tensor) -> Result<Tensor> {
        self.pow_(rhs)
    }

    /// Ternary select: take `on_true` where `self != 0`, else `on_false`.
    /// All three shapes and dtypes must be equal — there is no bool dtype.
    pub fn where_cond(&self, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor> {
        if self.dtype() != on_true.dtype() || self.dtype() != on_false.dtype() {
            return Err(Error::Dtype(format!(
                "where_cond needs one dtype: {:?}, {:?}, {:?}",
                self.dtype(),
                on_true.dtype(),
                on_false.dtype()
            )));
        }
        let expr = ScalarExpr::select(self.arg(0), on_true.arg(1), on_false.arg(2));
        Tensor::mapn(&self.graph, expr, &[self, on_true, on_false])
    }
}

// ---------------------------------------------------------------------------
// std::ops, all four owned x ref combinations, panicking on the Result
// ---------------------------------------------------------------------------

macro_rules! std_binop {
    ($trait:ident, $method:ident, $call:ident) => {
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$call(&self, &rhs).expect(concat!(stringify!($call), " failed"))
            }
        }
        impl std::ops::$trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$call(&self, rhs).expect(concat!(stringify!($call), " failed"))
            }
        }
        impl std::ops::$trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$call(self, &rhs).expect(concat!(stringify!($call), " failed"))
            }
        }
        impl std::ops::$trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$call(self, rhs).expect(concat!(stringify!($call), " failed"))
            }
        }
    };
}

std_binop!(Add, add, add);
std_binop!(Sub, sub, sub);
std_binop!(Mul, mul, mul);
std_binop!(Div, div, div);
std_binop!(Rem, rem, rem);

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(&self).expect("neg failed")
    }
}
impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        Tensor::neg(self).expect("neg failed")
    }
}

#[cfg(test)]
mod tests {
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::scalar::{BinOp, ScalarExpr, ScalarKind, UnOp};

    /// The op identity lives in the `ScalarExpr`, not in a node discriminant:
    /// `exp` and `log` differ only in one `UnOp`.
    #[test]
    fn exp_and_log_differ_only_in_the_scalar_expr() {
        let a = ScalarExpr::arg(0, Dtype::F32);
        let e = ScalarExpr::un(UnOp::Exp, a.clone());
        let l = ScalarExpr::un(UnOp::Log, a);
        assert_ne!(e, l);
        match (e.kind(), l.kind()) {
            (ScalarKind::Un { x: xa, .. }, ScalarKind::Un { x: xb, .. }) => assert_eq!(xa, xb),
            _ => panic!("expected two unaries"),
        }
    }

    /// The three exponentials are three nodes. They were all `UnOp::Exp`, so
    /// `approximate_exp(x)` and `exp(x)` hash-consed to one value and the
    /// accuracy contract had nowhere to live.
    #[test]
    fn the_three_exponentials_are_three_distinct_nodes() {
        let a = ScalarExpr::arg(0, Dtype::F32);
        let exact = ScalarExpr::un(UnOp::Exp, a.clone());
        let approx = ScalarExpr::un(UnOp::ApproximateExp, a.clone());
        let less = ScalarExpr::un(UnOp::LessApproximateExp, a);
        assert_ne!(exact, approx);
        assert_ne!(exact, less);
        assert_ne!(approx, less);
    }

    #[test]
    fn sqr_reads_the_operand_twice_in_one_expression() {
        let a = ScalarExpr::arg(0, Dtype::F32);
        let s = ScalarExpr::bin(BinOp::Mul, a.clone(), a);
        match s.kind() {
            ScalarKind::Bin { op, a, b } => {
                assert_eq!(*op, BinOp::Mul);
                assert_eq!(a, b);
            }
            _ => panic!("expected a binary"),
        }
    }
}
