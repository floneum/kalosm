//! The scalar-arith ops. The scalar becomes a `ScalarKind::Lit` when it is a
//! compile-time constant and a `ScalarKind::Uniform` when it is a runtime
//! value — `m.mul_scalar(lr)` with `lr: Scalar::Uniform(..)` produces a
//! `Uniform`, never a baked literal, which is trainer constraint 2.
//!
//! Every entry here is **one `Logical::Map`** carrying a `Lit`/`Uniform` leaf, not
//! a broadcast const tensor: `clamp` is a single `Min(Max(x, lo), hi)` node,
//! not two.

use fusor_ir::scalar::{BinOp, ScalarExpr};

use crate::Result;
use crate::tensor::{Scalar, Tensor};

impl Tensor {
    /// `x op s`, one `Map`.
    fn scalar_rhs(&self, op: BinOp, s: impl Into<Scalar>) -> Result<Tensor> {
        let e = s.into().expr(self.dtype());
        self.map1(ScalarExpr::bin(op, self.arg0(), e))
    }

    /// `s op x`, one `Map`.
    fn scalar_lhs(&self, op: BinOp, s: impl Into<Scalar>) -> Result<Tensor> {
        let e = s.into().expr(self.dtype());
        self.map1(ScalarExpr::bin(op, e, self.arg0()))
    }

    /// `x + s`.
    pub fn add_scalar(&self, rhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_rhs(BinOp::Add, rhs)
    }
    /// `x - s`.
    pub fn sub_scalar(&self, rhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_rhs(BinOp::Sub, rhs)
    }
    /// `s - x`.
    pub fn rsub_scalar(&self, lhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_lhs(BinOp::Sub, lhs)
    }
    /// `x * s`.
    pub fn mul_scalar(&self, rhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_rhs(BinOp::Mul, rhs)
    }
    /// `x / s`.
    pub fn div_scalar(&self, rhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_rhs(BinOp::Div, rhs)
    }
    /// `s / x`.
    pub fn rdiv_scalar(&self, lhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_lhs(BinOp::Div, lhs)
    }
    /// `x ^ s`.
    pub fn pow_scalar(&self, rhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_rhs(BinOp::Pow, rhs)
    }
    /// `max(x, s)`. The basis of `relu`.
    pub fn max_scalar(&self, rhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_rhs(BinOp::Max, rhs)
    }
    /// `min(x, s)`.
    pub fn min_scalar(&self, rhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_rhs(BinOp::Min, rhs)
    }
    /// `x % s`, truncated toward zero on every dtype.
    ///
    /// Unlike [`Tensor::rem`], this is not integer-only. Tensor-tensor
    /// remainder remains integer-only.
    pub fn rem_scalar(&self, rhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_rhs(BinOp::Rem, rhs)
    }
    /// `min(max(x, lo), hi)` as **one** `Map`, not two.
    pub fn clamp(&self, lo: impl Into<Scalar>, hi: impl Into<Scalar>) -> Result<Tensor> {
        let dt = self.dtype();
        let lo = lo.into().expr(dt);
        let hi = hi.into().expr(dt);
        let inner = ScalarExpr::bin(BinOp::Max, self.arg0(), lo);
        self.map1(ScalarExpr::bin(BinOp::Min, inner, hi))
    }
}
