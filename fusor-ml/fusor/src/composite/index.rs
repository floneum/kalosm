//! Indexing operations for tensors.
//!
//! This module provides PyTorch-style tensor indexing via the `i()` method.
//! Example: `tensor.i((.., 0, ..))` to select a specific index along one dimension.

use crate::gpu::DataType;
use crate::{ConcreteTensor, SimdElement, Tensor};
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

/// Helper enum for flexible indexing (range or single index)
#[derive(Clone)]
pub enum IndexOp {
    Full,
    Range(Range<usize>),
    RangeTo(usize),
    RangeFrom(usize),
    Index(usize),
}

impl From<RangeFull> for IndexOp {
    fn from(_: RangeFull) -> Self {
        IndexOp::Full
    }
}

impl From<Range<usize>> for IndexOp {
    fn from(r: Range<usize>) -> Self {
        IndexOp::Range(r)
    }
}

impl From<RangeTo<usize>> for IndexOp {
    fn from(r: RangeTo<usize>) -> Self {
        IndexOp::RangeTo(r.end)
    }
}

impl From<RangeFrom<usize>> for IndexOp {
    fn from(r: RangeFrom<usize>) -> Self {
        IndexOp::RangeFrom(r.start)
    }
}

impl From<usize> for IndexOp {
    fn from(i: usize) -> Self {
        IndexOp::Index(i)
    }
}

impl IndexOp {
    fn to_range(&self, dim_size: usize) -> Range<usize> {
        match self {
            IndexOp::Full => 0..dim_size,
            IndexOp::Range(r) => r.clone(),
            IndexOp::RangeTo(end) => 0..*end,
            IndexOp::RangeFrom(start) => *start..dim_size,
            IndexOp::Index(i) => *i..(*i + 1),
        }
    }

    fn removes_dim(&self) -> bool {
        matches!(self, IndexOp::Index(_))
    }
}

/// Converts rank-specific index tuples into tensor indexing operations.
pub trait TensorIndex<const R: usize, D>
where
    D: SimdElement + DataType + Default,
{
    /// Tensor produced by the indexing operation.
    type Output;

    /// Apply this index to `tensor`.
    fn index(self, tensor: &Tensor<R, D, ConcreteTensor<D, R>>) -> Self::Output;
}

impl<const R: usize, D> Tensor<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + Default,
{
    /// Index into a tensor, reducing exactly one indexed dimension.
    pub fn i<I>(&self, index: I) -> I::Output
    where
        I: TensorIndex<R, D>,
    {
        index.index(self)
    }
}

fn removed_dim<const R: usize>(removes: [bool; R]) -> usize {
    let num_removes = removes.iter().filter(|&&removed| removed).count();
    assert!(
        num_removes == 1,
        "i() expects exactly one index (not range) to reduce rank, got {} indices",
        num_removes
    );
    removes
        .iter()
        .position(|&removed| removed)
        .expect("checked exactly one removed dimension")
}

impl<D, I1, I2> TensorIndex<2, D> for (I1, I2)
where
    D: SimdElement + DataType + Default,
    I1: Into<IndexOp>,
    I2: Into<IndexOp>,
    crate::cpu::ConcreteTensor<D, 2>: crate::cpu::LastRank<1, D>,
    crate::gpu::Tensor<2, D>: crate::gpu::LastRank<1, D>,
{
    type Output = Tensor<1, D>;

    fn index(self, tensor: &Tensor<2, D, ConcreteTensor<D, 2>>) -> Self::Output {
        let (i1, i2) = self;
        let i1 = i1.into();
        let i2 = i2.into();
        let shape = tensor.shape();
        let slices = [i1.to_range(shape[0]), i2.to_range(shape[1])];
        let sliced = tensor.slice(slices).to_concrete();
        let dim = removed_dim([i1.removes_dim(), i2.removes_dim()]);
        sliced.squeeze::<1>(dim).to_concrete()
    }
}

impl<D, I1, I2, I3> TensorIndex<3, D> for (I1, I2, I3)
where
    D: SimdElement + DataType + Default,
    I1: Into<IndexOp>,
    I2: Into<IndexOp>,
    I3: Into<IndexOp>,
    crate::cpu::ConcreteTensor<D, 3>: crate::cpu::LastRank<2, D>,
    crate::gpu::Tensor<3, D>: crate::gpu::LastRank<2, D>,
{
    type Output = Tensor<2, D>;

    fn index(self, tensor: &Tensor<3, D, ConcreteTensor<D, 3>>) -> Self::Output {
        let (i1, i2, i3) = self;
        let i1 = i1.into();
        let i2 = i2.into();
        let i3 = i3.into();
        let shape = tensor.shape();
        let slices = [
            i1.to_range(shape[0]),
            i2.to_range(shape[1]),
            i3.to_range(shape[2]),
        ];
        let sliced = tensor.slice(slices).to_concrete();
        let dim = removed_dim([i1.removes_dim(), i2.removes_dim(), i3.removes_dim()]);
        sliced.squeeze::<2>(dim).to_concrete()
    }
}

impl<D, I1, I2, I3, I4> TensorIndex<4, D> for (I1, I2, I3, I4)
where
    D: SimdElement + DataType + Default,
    I1: Into<IndexOp>,
    I2: Into<IndexOp>,
    I3: Into<IndexOp>,
    I4: Into<IndexOp>,
    crate::cpu::ConcreteTensor<D, 4>: crate::cpu::LastRank<3, D>,
    crate::gpu::Tensor<4, D>: crate::gpu::LastRank<3, D>,
{
    type Output = Tensor<3, D>;

    fn index(self, tensor: &Tensor<4, D, ConcreteTensor<D, 4>>) -> Self::Output {
        let (i1, i2, i3, i4) = self;
        let i1 = i1.into();
        let i2 = i2.into();
        let i3 = i3.into();
        let i4 = i4.into();
        let shape = tensor.shape();
        let slices = [
            i1.to_range(shape[0]),
            i2.to_range(shape[1]),
            i3.to_range(shape[2]),
            i4.to_range(shape[3]),
        ];
        let sliced = tensor.slice(slices).to_concrete();
        let dim = removed_dim([
            i1.removes_dim(),
            i2.removes_dim(),
            i3.removes_dim(),
            i4.removes_dim(),
        ]);
        sliced.squeeze::<3>(dim).to_concrete()
    }
}
