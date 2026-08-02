//! The reductions. `sum`, `max`, `min` and `product` are each one `L0::Fold`
//! at a different single-slot [`Carrier`]; `mean` and `var` are compositions of those and
//! a `Map`. Nothing here chooses a reduction strategy — `fold_split` is the
//! single rule that turns any of them into a two-stage reduction when the
//! extractor decides the axis is long enough to pay for it.
//!
//! Owned by W12.

use fusor2_ir::carrier::Carrier;
use fusor2_ir::ir::level0::{L0, TiePolicy};
use fusor2_ir::scalar::BinOp;
use fusor2_ir::shape::Dim;

use crate::tensor::{Scalar, Tensor};
use crate::{Error, Result};

impl Tensor {
    /// One `L0::Fold` over `axis` with a scalar carrier.
    ///
    /// The accumulator is `dtype.compute_dtype()`, so an f16 reduction
    /// accumulates — and therefore *results* — in f32. That is the
    /// mixed-precision story the architecture asks for: `acc` is carried
    /// separately from operand dtype, and narrowing back is an explicit
    /// `cast` the caller writes.
    fn fold(&self, op: BinOp, tie: Option<TiePolicy>, axis: usize) -> Result<Tensor> {
        self.require_dense("reduction")?;
        self.check_axis(axis, "reduction")?;
        let acc = self.dtype().compute_dtype();
        let ident = Carrier::binop_identity(op, acc)
            .ok_or_else(|| Error::Dtype(format!("{op:?} has no identity in {acc:?}")))?;
        let mut carrier = Carrier::binop(op, ident, acc);
        carrier.tie = tie;
        self.emit_here(L0::Fold {
            carrier,
            axis: axis as u32,
            acc,
            ins: smallvec::smallvec![self.id],
        })
    }

    /// One `L0::Fold` over `axis` at an **arbitrary** carrier.
    ///
    /// The general form the fold laws mint: a multi-slot accumulator whose
    /// `lanes` are appended to the output shape as a trailing axis, so slot `k`
    /// reads back as an ordinary `Restride`. `sum`/`max`/`min`/`product` above
    /// are this with a one-slot binop carrier.
    ///
    /// The accumulator dtype is `dtype.compute_dtype()`, matching the scalar
    /// reductions; the carrier's identities must be values of it, which
    /// `verify_l0` clause 3 checks along with identity closure.
    pub fn fold_carrier(&self, carrier: Carrier, axis: usize) -> Result<Tensor> {
        self.require_dense("reduction")?;
        self.check_axis(axis, "reduction")?;
        let acc = self.dtype().compute_dtype();
        self.emit_here(L0::Fold {
            carrier,
            axis: axis as u32,
            acc,
            ins: smallvec::smallvec![self.id],
        })
    }

    /// Sum over `axis`, dropping it.
    pub fn sum(&self, axis: usize) -> Result<Tensor> {
        self.fold(BinOp::Add, None, axis)
    }
    /// Product over `axis`, dropping it.
    pub fn product(&self, axis: usize) -> Result<Tensor> {
        self.fold(BinOp::Mul, None, axis)
    }
    /// Maximum over `axis`, ties split evenly on the way back.
    pub fn max(&self, axis: usize) -> Result<Tensor> {
        self.fold(BinOp::Max, Some(TiePolicy::SplitEvenly), axis)
    }
    /// Minimum over `axis`, ties split evenly on the way back.
    pub fn min(&self, axis: usize) -> Result<Tensor> {
        self.fold(BinOp::Min, Some(TiePolicy::SplitEvenly), axis)
    }
    /// Maximum with an explicit tie policy. Parity with a reference trainer
    /// is a declaration, not an accident.
    pub fn max_with_tie(&self, axis: usize, tie: TiePolicy) -> Result<Tensor> {
        self.fold(BinOp::Max, Some(tie), axis)
    }
    /// Minimum with an explicit tie policy.
    pub fn min_with_tie(&self, axis: usize, tie: TiePolicy) -> Result<Tensor> {
        self.fold(BinOp::Min, Some(tie), axis)
    }

    /// Sum over `axis`, keeping it at extent 1.
    pub fn sum_keepdim(&self, axis: usize) -> Result<Tensor> {
        self.sum(axis)?.unsqueeze(axis)
    }
    /// Product over `axis`, keeping it at extent 1.
    pub fn product_keepdim(&self, axis: usize) -> Result<Tensor> {
        self.product(axis)?.unsqueeze(axis)
    }
    /// Maximum over `axis`, keeping it at extent 1.
    pub fn max_keepdim(&self, axis: usize) -> Result<Tensor> {
        self.max(axis)?.unsqueeze(axis)
    }
    /// Minimum over `axis`, keeping it at extent 1.
    pub fn min_keepdim(&self, axis: usize) -> Result<Tensor> {
        self.min(axis)?.unsqueeze(axis)
    }

    /// The reciprocal of the extent of `axis`, as a literal when the extent is
    /// `Const` and as a **uniform divisor** when it is `Sym`. A symbolic
    /// sequence length must not bake itself into a shader.
    fn axis_divisor(&self, axis: usize) -> Result<Divisor> {
        Ok(match self.dim(axis) {
            Dim::Const(0) => return Err(Error::Shape(format!("mean over empty axis {axis}"))),
            Dim::Const(n) => Divisor::Reciprocal(1.0 / n as f32),
            Dim::Sym(s) => Divisor::Uniform(s),
        })
    }

    /// Arithmetic mean over `axis`.
    pub fn mean(&self, axis: usize) -> Result<Tensor> {
        self.check_axis(axis, "mean")?;
        let d = self.axis_divisor(axis)?;
        let s = self.sum(axis)?;
        d.apply(&s)
    }

    /// Arithmetic mean over `axis`, keeping it at extent 1.
    pub fn mean_keepdim(&self, axis: usize) -> Result<Tensor> {
        self.mean(axis)?.unsqueeze(axis)
    }

    /// Biased variance, `mean(x^2) - mean(x)^2`.
    pub fn var(&self, axis: usize) -> Result<Tensor> {
        let m2 = self.sqr()?.mean(axis)?;
        let m = self.mean(axis)?;
        m2.sub(&m.sqr()?)
    }

    /// Biased variance, keeping `axis` at extent 1.
    pub fn var_keepdim(&self, axis: usize) -> Result<Tensor> {
        self.var(axis)?.unsqueeze(axis)
    }

    /// Reference spelling of [`Tensor::var`].
    pub fn variance(&self, axis: usize) -> Result<Tensor> {
        self.var(axis)
    }

    // -- whole-tensor folds ---------------------------------------------------

    /// Sum every element into a rank-0 value.
    pub fn sum_all(&self) -> Result<Tensor> {
        self.flatten_all()?.sum(0)
    }
    /// Maximum of every element, rank 0.
    pub fn max_all(&self) -> Result<Tensor> {
        self.flatten_all()?.max(0)
    }
    /// Minimum of every element, rank 0.
    pub fn min_all(&self) -> Result<Tensor> {
        self.flatten_all()?.min(0)
    }
    /// Product of every element, rank 0.
    pub fn product_all(&self) -> Result<Tensor> {
        self.flatten_all()?.product(0)
    }
    /// Mean of every element, rank 0.
    pub fn mean_all(&self) -> Result<Tensor> {
        self.flatten_all()?.mean(0)
    }

    // -- derived reductions ----------------------------------------------------

    /// `1` where any element along `axis` is nonzero.
    pub fn any(&self, axis: usize) -> Result<Tensor> {
        self.ne_scalar(Scalar::Lit(crate::tensor::splat_zero(self.dtype())))?
            .max(axis)
    }

    /// `1` only where every element along `axis` is nonzero.
    pub fn all(&self, axis: usize) -> Result<Tensor> {
        self.ne_scalar(Scalar::Lit(crate::tensor::splat_zero(self.dtype())))?
            .min(axis)
    }

    /// Count of nonzero elements along `axis`, in the operand dtype.
    pub fn count_nonzero(&self, axis: usize) -> Result<Tensor> {
        self.ne_scalar(Scalar::Lit(crate::tensor::splat_zero(self.dtype())))?
            .sum(axis)
    }

    /// Euclidean norm along `axis`.
    pub fn norm(&self, axis: usize) -> Result<Tensor> {
        self.sqr()?.sum(axis)?.sqrt()
    }

    /// Index of the first maximum along `axis`, as `U32`.
    ///
    /// L0 has no index-carrying fold, so this is the honest composition:
    /// broadcast the extremum back, replace every non-extremal position with
    /// the `Min` identity, and fold `Min` over `IndexOf(axis)`. Exact for
    /// extents below `2^24` on an f32 tensor, where the index stops being
    /// representable.
    pub fn arg_max(&self, axis: usize) -> Result<Tensor> {
        self.arg_extremum(axis, BinOp::Max)
    }

    /// Index of the first minimum along `axis`, as `U32`.
    pub fn arg_min(&self, axis: usize) -> Result<Tensor> {
        self.arg_extremum(axis, BinOp::Min)
    }

    fn arg_extremum(&self, axis: usize, which: BinOp) -> Result<Tensor> {
        use fusor2_ir::scalar::{CmpOp, ScalarExpr};

        self.require_dense("arg_extremum")?;
        self.check_axis(axis, "arg_extremum")?;
        let dt = self.dtype();
        let sentinel = Carrier::binop_identity(BinOp::Min, dt)
            .ok_or_else(|| Error::Dtype(format!("no Min identity for {dt:?}")))?;

        let ext = self
            .fold(which, Some(TiePolicy::FirstWins), axis)?
            .unsqueeze(axis)?;
        let ext = ext.broadcast_as(&self.shape())?;

        let idx = ScalarExpr::cast(dt, ScalarExpr::index_of(axis as u32));
        let hit = ScalarExpr::cmp(CmpOp::Eq, self.arg(0), ext.arg(1));
        let body = ScalarExpr::select(hit, idx, ScalarExpr::lit(sentinel));
        let masked = Tensor::mapn(&self.graph, body, &[self, &ext])?;
        masked
            .fold(BinOp::Min, Some(TiePolicy::FirstWins), axis)?
            .cast(fusor2_ir::dtype::Dtype::U32)
    }
}

/// How `mean` divides: by a literal reciprocal, or by a runtime uniform.
enum Divisor {
    Reciprocal(f32),
    Uniform(fusor2_ir::shape::SymId),
}

impl Divisor {
    fn apply(self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Reciprocal(r) => x.mul_scalar(r),
            Self::Uniform(s) => x.div_scalar(Scalar::Uniform(s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use fusor2_ir::carrier::{Carrier, probes_for};
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::ir::level0::TiePolicy;
    use fusor2_ir::scalar::BinOp;

    #[test]
    fn every_reduction_is_one_identity_closed_scalar_carrier() {
        for op in [BinOp::Add, BinOp::Mul, BinOp::Max, BinOp::Min] {
            for dt in [Dtype::F32, Dtype::F16, Dtype::U32, Dtype::I32] {
                let acc = dt.compute_dtype();
                let ident = Carrier::binop_identity(op, acc)
                    .unwrap_or_else(|| panic!("{op:?} has no identity in {acc:?}"));
                let c = Carrier::binop(op, ident, acc);
                assert_eq!(c.width(), 1);
                assert_eq!(c.kind(), Some(op));
                assert!(c.identity_closed(probes_for(acc)), "{op:?} in {acc:?}");
            }
        }
    }

    /// The tie policy rides on the carrier and is part of the node key, so two
    /// maxima with different policies stay distinct nodes.
    #[test]
    fn the_tie_policy_is_part_of_the_carrier() {
        let ident = Carrier::binop_identity(BinOp::Max, Dtype::F32).unwrap();
        let split = Carrier::binop(BinOp::Max, ident, Dtype::F32).with_tie(TiePolicy::SplitEvenly);
        let first = Carrier::binop(BinOp::Max, ident, Dtype::F32).with_tie(TiePolicy::FirstWins);
        assert_eq!(split.tie, Some(TiePolicy::SplitEvenly));
        assert_ne!(split, first);
        assert_eq!(split.merge, first.merge, "only the adjoint attribute differs");
    }

    #[test]
    fn f16_reductions_accumulate_in_f32() {
        assert_eq!(Dtype::F16.compute_dtype(), Dtype::F32);
        assert_eq!(Dtype::BF16.compute_dtype(), Dtype::F32);
        assert_eq!(Dtype::F32.compute_dtype(), Dtype::F32);
        assert_eq!(Dtype::U32.compute_dtype(), Dtype::U32);
    }
}
