//! The preserved alias surface: the names the reference exposes that are thin
//! wrappers over a primitive here. Kept so callers do not have to be
//! rewritten.
//!
//! **Thin forwarders only.** Nothing in this file mints a node, so every alias
//! is structurally identical to its target and hash-conses onto it.
//!
//! (`softmax_slow*` lives in W13's `composite/normalization.rs`, not here.)
//!
//! Owned by W12.

use crate::tensor::{Scalar, Tensor};
use crate::Result;

impl Tensor {
    /// `more than` — the fusor-core-compatible spelling of
    /// [`Tensor::gt_scalar`].
    pub fn mt(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.gt_scalar(s)
    }
    /// `more than or equal` — [`Tensor::gte_scalar`].
    pub fn mte(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.gte_scalar(s)
    }

    /// [`Tensor::eq_scalar`].
    pub fn eq(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.eq_scalar(s)
    }
    /// [`Tensor::ne_scalar`].
    pub fn ne(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.ne_scalar(s)
    }
    /// [`Tensor::lt_scalar`].
    pub fn lt(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.lt_scalar(s)
    }
    /// [`Tensor::lte_scalar`].
    pub fn lte(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.lte_scalar(s)
    }
    /// [`Tensor::gt_scalar`].
    pub fn gt(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.gt_scalar(s)
    }
    /// [`Tensor::gte_scalar`].
    pub fn gte(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.gte_scalar(s)
    }

    /// [`Tensor::pow_scalar`].
    pub fn pow_elementwise(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.pow_scalar(s)
    }
    /// [`Tensor::max_scalar`]. The reference's `relu` is `max_elementwise(0)`.
    pub fn max_elementwise(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.max_scalar(s)
    }
    /// [`Tensor::min_scalar`].
    pub fn min_elementwise(&self, s: impl Into<Scalar>) -> Result<Tensor> {
        self.min_scalar(s)
    }

    /// [`Tensor::matmul`].
    pub fn mat_mul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.matmul(rhs)
    }
    /// [`Tensor::matmul_t`].
    pub fn mt_matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.matmul_t(rhs)
    }
    /// [`Tensor::matmul_t`], the reference's spelling.
    pub fn mat_mul_transposed_rhs(&self, rhs: &Tensor) -> Result<Tensor> {
        self.matmul_t(rhs)
    }
}
