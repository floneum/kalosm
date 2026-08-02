//! The scalar-arith ops. The scalar becomes a `ScalarKind::Lit` when it is a
//! compile-time constant and a `ScalarKind::Uniform` when it is a runtime
//! value — `m.mul_scalar(lr)` with `lr: Scalar::Uniform(..)` produces a
//! `Uniform`, never a baked literal, which is trainer constraint 2.
//!
//! Every entry here is **one `L0::Map`** carrying a `Lit`/`Uniform` leaf, not
//! a broadcast const tensor: `clamp` is a single `Min(Max(x, lo), hi)` node,
//! not two.
//!
//! Owned by W12.

use fusor2_ir::scalar::{BinOp, ScalarExpr};

use crate::tensor::{Scalar, Tensor};
use crate::Result;

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
    /// **Not integer-only**, unlike [`Tensor::rem`]. The reference defines the
    /// scalar remainder over `f32`, `f16` and `u32`
    /// (`element_wise.rs:94`, `NaryOp::RRemConst`), and both emitters already
    /// compute a float remainder — the CPU as `x - trunc(x / y) * y` and the
    /// GPU as WGSL `%`, which is the same truncated form Rust's `%` uses. The
    /// tensor-tensor spelling stays integer-only because that is where the
    /// reference's SIMD coverage stops, and `dtypes::rem_is_u32_only` pins it.
    pub fn rem_scalar(&self, rhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_rhs(BinOp::Rem, rhs)
    }
    /// `s % x`. See [`Tensor::rem_scalar`].
    pub fn rrem_scalar(&self, lhs: impl Into<Scalar>) -> Result<Tensor> {
        self.scalar_lhs(BinOp::Rem, lhs)
    }

    /// `min(max(x, lo), hi)` as **one** `Map`, not two.
    pub fn clamp(&self, lo: impl Into<Scalar>, hi: impl Into<Scalar>) -> Result<Tensor> {
        let dt = self.dtype();
        let lo = lo.into().expr(dt);
        let hi = hi.into().expr(dt);
        let inner = ScalarExpr::bin(BinOp::Max, self.arg0(), lo);
        self.map1(ScalarExpr::bin(BinOp::Min, inner, hi))
    }

    /// Multiply by a rank-0 runtime value carried as a tensor. Prefer
    /// `mul_scalar(Scalar::Uniform(..))`, which needs no operand at all.
    pub fn mul_uniform(&self, uniform: &Tensor) -> Result<Tensor> {
        self.mul_(uniform)
    }
}

#[cfg(test)]
mod tests {
    use fusor2_ir::dtype::{Dtype, Splat};
    use fusor2_ir::scalar::{BinOp, ScalarExpr, ScalarKind};
    use fusor2_ir::shape::SymId;

    use crate::tensor::Scalar;

    /// A literal scalar is a `Lit`; a runtime scalar is a `Uniform`, and the
    /// expression's structural hash does not depend on the bound value.
    #[test]
    fn lit_versus_uniform() {
        let arg = ScalarExpr::arg(0, Dtype::F32);
        let lit = ScalarExpr::bin(BinOp::Mul, arg.clone(), Scalar::from(2.0f32).expr(Dtype::F32));
        assert!(matches!(
            lit.kind(),
            ScalarKind::Bin { b, .. } if matches!(b.kind(), ScalarKind::Lit(_))
        ));

        let u = ScalarExpr::bin(
            BinOp::Mul,
            arg.clone(),
            Scalar::Uniform(SymId(4)).expr(Dtype::F32),
        );
        assert!(matches!(
            u.kind(),
            ScalarKind::Bin { b, .. } if matches!(b.kind(), ScalarKind::Uniform(SymId(4)))
        ));

        // Two different literals hash differently; the uniform is one node
        // regardless of what value binding 0 later carries.
        let other = ScalarExpr::bin(BinOp::Mul, arg, Scalar::from(3.0f32).expr(Dtype::F32));
        assert_ne!(lit.structural_hash(), other.structural_hash());
        assert_eq!(
            u.structural_hash(),
            ScalarExpr::bin(
                BinOp::Mul,
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::uniform(SymId(4), Dtype::F32)
            )
            .structural_hash()
        );
    }

    #[test]
    fn clamp_is_one_expression() {
        let arg = ScalarExpr::arg(0, Dtype::F32);
        let inner = ScalarExpr::bin(BinOp::Max, arg, ScalarExpr::lit(Splat::F32(0.0)));
        let e = ScalarExpr::bin(BinOp::Min, inner, ScalarExpr::lit(Splat::F32(6.0)));
        match e.kind() {
            ScalarKind::Bin { op, a, .. } => {
                assert_eq!(*op, BinOp::Min);
                assert!(matches!(a.kind(), ScalarKind::Bin { op: BinOp::Max, .. }));
            }
            _ => panic!("expected a binary"),
        }
    }

    #[test]
    fn a_literal_is_retyped_to_the_operand_dtype() {
        let e = Scalar::from(2.0f32).expr(Dtype::U32);
        assert_eq!(e.dtype(), Dtype::U32);
        assert!(matches!(e.kind(), ScalarKind::Lit(l) if l.0 == Splat::U32(2)));
    }
}
