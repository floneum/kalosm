//! The reductions. `sum`, `max`, `min` and `product` are each one `Logical::Fold`
//! at a different single-slot [`Carrier`]; `mean` and `var` are compositions of those and
//! a `Map`. Nothing here chooses a reduction strategy — `fold_split` is the
//! single rule that turns any of them into a two-stage reduction when the
//! extractor decides the axis is long enough to pay for it.

use fusor2_ir::carrier::Carrier;
use fusor2_ir::ir::logical::{Logical, TiePolicy};
use fusor2_ir::scalar::BinOp;
use fusor2_ir::shape::Dim;

use crate::tensor::{Scalar, Tensor};
use crate::{Error, Result};

impl Tensor {
    /// One `Logical::Fold` over `axis` with a scalar carrier.
    ///
    /// The accumulator is `dtype.compute_dtype()`, so an f16 reduction
    /// accumulates — and therefore results — in f32; narrowing back is an
    /// explicit `cast` the caller writes.
    fn fold(&self, op: BinOp, tie: Option<TiePolicy>, axis: usize) -> Result<Tensor> {
        self.require_dense("reduction")?;
        self.check_axis(axis, "reduction")?;
        let acc = self.dtype().compute_dtype();
        let ident = Carrier::binop_identity(op, acc)
            .ok_or_else(|| Error::Dtype(format!("{op:?} has no identity in {acc:?}")))?;
        let mut carrier = Carrier::binop(op, ident, acc);
        carrier.tie = tie;
        self.emit_here(Logical::Fold {
            carrier,
            axis: axis as u32,
            acc,
            ins: smallvec::smallvec![self.id],
        })
    }

    /// One `Logical::Fold` over `axis` at an **arbitrary** carrier.
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
        self.emit_here(Logical::Fold {
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
    /// Maximum with an explicit tie policy.
    pub fn max_with_tie(&self, axis: usize, tie: TiePolicy) -> Result<Tensor> {
        self.fold(BinOp::Max, Some(tie), axis)
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

    /// Sum every element into a rank-0 value.
    pub fn sum_all(&self) -> Result<Tensor> {
        self.flatten_all()?.sum(0)
    }
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
    /// Logical has no index-carrying fold, so this is a composition:
    /// broadcast the extremum back, replace every non-extremal position with
    /// the `Min` identity, and fold `Min` over `IndexOf(axis)`. Exact for
    /// extents below `2^24` on an f32 tensor, where the index stops being
    /// representable.
    pub fn arg_max(&self, axis: usize) -> Result<Tensor> {
        self.arg_extremum(axis, BinOp::Max)
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
