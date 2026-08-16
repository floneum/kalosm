//! The 12 comparisons. Results are 1.0/0.0 in the operand dtype; there is no
//! boolean dtype at Logical.

use fusor2_ir::scalar::{CmpOp, ScalarExpr};

use crate::tensor::{Scalar, Tensor};
use crate::{Error, Result};

impl Tensor {
    /// `cmp(x, s)`, one `Map`.
    fn cmp_scalar(&self, op: CmpOp, s: impl Into<Scalar>) -> Result<Tensor> {
        let e = s.into().expr(self.dtype());
        self.map1(ScalarExpr::cmp(op, self.arg0(), e))
    }

    /// `cmp(a, b)`, one `Map`; shapes and dtypes must already match.
    fn cmp_tensor(&self, op: CmpOp, rhs: &Tensor, what: &str) -> Result<Tensor> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::Dtype(format!(
                "{what} operands differ in dtype: {:?} vs {:?}",
                self.dtype(),
                rhs.dtype()
            )));
        }
        let expr = ScalarExpr::cmp(op, self.arg(0), rhs.arg(1));
        Tensor::mapn(&self.graph, expr, &[self, rhs])
    }

    /// `1` where `x == s`, else `0`, in `x`'s dtype.
    pub fn eq_scalar(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.cmp_scalar(CmpOp::Eq, s)
    }
    /// `1` where `x != s`.
    pub fn ne_scalar(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.cmp_scalar(CmpOp::Ne, s)
    }
    /// `1` where `x < s`.
    pub fn lt_scalar(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.cmp_scalar(CmpOp::Lt, s)
    }
    /// `1` where `x <= s`.
    pub fn lte_scalar(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.cmp_scalar(CmpOp::Le, s)
    }
    /// `1` where `x > s`.
    pub fn gt_scalar(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.cmp_scalar(CmpOp::Gt, s)
    }
    /// `1` where `x >= s`.
    pub fn gte_scalar(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.cmp_scalar(CmpOp::Ge, s)
    }
    /// `1` where `a == b`.
    pub fn eq_tensor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.cmp_tensor(CmpOp::Eq, rhs, "eq_tensor")
    }
    /// `1` where `a != b`.
    pub fn ne_tensor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.cmp_tensor(CmpOp::Ne, rhs, "ne_tensor")
    }
    /// `1` where `a < b`.
    pub fn lt_tensor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.cmp_tensor(CmpOp::Lt, rhs, "lt_tensor")
    }
    /// `1` where `a <= b`.
    pub fn lte_tensor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.cmp_tensor(CmpOp::Le, rhs, "lte_tensor")
    }
    /// `1` where `a > b`.
    pub fn gt_tensor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.cmp_tensor(CmpOp::Gt, rhs, "gt_tensor")
    }
    /// `1` where `a >= b`.
    pub fn gte_tensor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.cmp_tensor(CmpOp::Ge, rhs, "gte_tensor")
    }
}

#[cfg(test)]
mod tests {
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::scalar::{CmpOp, ScalarExpr, ScalarKind};

    /// A comparison's result carries the *operand* dtype, not a bool.
    #[test]
    fn comparisons_stay_in_the_operand_dtype() {
        for dt in [Dtype::F32, Dtype::F16, Dtype::U32, Dtype::I32] {
            let e = ScalarExpr::cmp(CmpOp::Lt, ScalarExpr::arg(0, dt), ScalarExpr::arg(1, dt));
            assert_eq!(e.dtype(), dt);
        }
    }

    /// `Ne` is a primitive.
    #[test]
    fn ne_is_one_node() {
        let e = ScalarExpr::cmp(
            CmpOp::Ne,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::lit(fusor2_ir::dtype::Splat::F32(0.0)),
        );
        assert!(matches!(e.kind(), ScalarKind::Cmp { op: CmpOp::Ne, .. }));
    }
}
