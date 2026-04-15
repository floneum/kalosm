//! Indexing operations for tensors.
//!
//! This module provides PyTorch-style tensor indexing via the `i()` method.
//! Example: `tensor.i((.., 0, ..))` to select a specific index along one dimension.

use crate::{ConcreteTensor, SimdElement, Tensor};
use fusor_core::DataType;
use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

// Note: TensorIndex traits are complex and rank-dependent.
// We provide direct implementations for common use cases.

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

// Implement i() for 2D tensors
impl<D> Tensor<2, D, ConcreteTensor<D, 2>>
where
    D: SimdElement + DataType + Default,
{
    /// Index into a 2D tensor. Returns a 1D tensor when one index is specified,
    /// or a 2D tensor when ranges are used.
    pub fn i<I1, I2>(&self, (i1, i2): (I1, I2)) -> Tensor<1, D>
    where
        I1: Into<IndexOp>,
        I2: Into<IndexOp>,
        fusor_cpu::ConcreteTensor<D, 2>: fusor_cpu::LastRank<1, D>,
        fusor_core::Tensor<2, D>: fusor_core::LastRank<1, D>,
    {
        let i1 = i1.into();
        let i2 = i2.into();
        let shape = self.shape();

        let slices = [i1.to_range(shape[0]), i2.to_range(shape[1])];

        let sliced = self.slice(slices).to_concrete();

        // Squeeze dimensions that were indexed with a single value
        if i2.removes_dim() {
            sliced.squeeze::<1>(1).to_concrete()
        } else if i1.removes_dim() {
            sliced.squeeze::<1>(0).to_concrete()
        } else {
            panic!("i() on 2D tensor with two ranges should return 2D tensor, use slice() instead")
        }
    }
}

// Implement i() for 3D tensors
impl<D> Tensor<3, D, ConcreteTensor<D, 3>>
where
    D: SimdElement + DataType + Default,
{
    /// Index into a 3D tensor.
    pub fn i<I1, I2, I3>(&self, (i1, i2, i3): (I1, I2, I3)) -> Tensor<2, D>
    where
        I1: Into<IndexOp>,
        I2: Into<IndexOp>,
        I3: Into<IndexOp>,
        fusor_cpu::ConcreteTensor<D, 3>: fusor_cpu::LastRank<2, D>,
        fusor_core::Tensor<3, D>: fusor_core::LastRank<2, D>,
    {
        let i1 = i1.into();
        let i2 = i2.into();
        let i3 = i3.into();
        let shape = self.shape();

        let slices = [
            i1.to_range(shape[0]),
            i2.to_range(shape[1]),
            i3.to_range(shape[2]),
        ];

        let sliced = self.slice(slices).to_concrete();

        // Count how many dimensions are being removed
        let removes = [i1.removes_dim(), i2.removes_dim(), i3.removes_dim()];
        let num_removes: usize = removes.iter().filter(|&&x| x).count();

        if num_removes != 1 {
            panic!(
                "i() on 3D tensor expects exactly one index (not range) to reduce to 2D, got {} indices",
                num_removes
            );
        }

        // Squeeze from last to first to keep indices valid
        if removes[2] {
            sliced.squeeze::<2>(2).to_concrete()
        } else if removes[1] {
            sliced.squeeze::<2>(1).to_concrete()
        } else {
            sliced.squeeze::<2>(0).to_concrete()
        }
    }
}

// Implement i() for 4D tensors
impl<D> Tensor<4, D, ConcreteTensor<D, 4>>
where
    D: SimdElement + DataType + Default,
{
    /// Index into a 4D tensor.
    pub fn i<I1, I2, I3, I4>(&self, (i1, i2, i3, i4): (I1, I2, I3, I4)) -> Tensor<3, D>
    where
        I1: Into<IndexOp>,
        I2: Into<IndexOp>,
        I3: Into<IndexOp>,
        I4: Into<IndexOp>,
        fusor_cpu::ConcreteTensor<D, 4>: fusor_cpu::LastRank<3, D>,
        fusor_core::Tensor<4, D>: fusor_core::LastRank<3, D>,
    {
        let i1 = i1.into();
        let i2 = i2.into();
        let i3 = i3.into();
        let i4 = i4.into();
        let shape = self.shape();

        let slices = [
            i1.to_range(shape[0]),
            i2.to_range(shape[1]),
            i3.to_range(shape[2]),
            i4.to_range(shape[3]),
        ];

        let sliced = self.slice(slices).to_concrete();

        let removes = [
            i1.removes_dim(),
            i2.removes_dim(),
            i3.removes_dim(),
            i4.removes_dim(),
        ];
        let num_removes: usize = removes.iter().filter(|&&x| x).count();

        if num_removes != 1 {
            panic!(
                "i() on 4D tensor expects exactly one index (not range) to reduce to 3D, got {} indices",
                num_removes
            );
        }

        if removes[3] {
            sliced.squeeze::<3>(3).to_concrete()
        } else if removes[2] {
            sliced.squeeze::<3>(2).to_concrete()
        } else if removes[1] {
            sliced.squeeze::<3>(1).to_concrete()
        } else {
            sliced.squeeze::<3>(0).to_concrete()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_index_2d_row() {
        // Create a 2D tensor [[1, 2], [3, 4], [5, 6]]
        let tensor: Tensor<2, f32, ConcreteTensor<f32, 2>> = Tensor::Cpu(
            fusor_cpu::Tensor::from_slice([3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );

        // Select row 1: [3, 4]
        let indexed = tensor.i((1, ..));
        let slice = indexed.as_slice().await.unwrap();

        assert_eq!(slice[[0]], 3.0);
        assert_eq!(slice[[1]], 4.0);
    }

    #[tokio::test]
    async fn test_index_2d_col() {
        // Create a 2D tensor [[1, 2], [3, 4], [5, 6]]
        let tensor: Tensor<2, f32, ConcreteTensor<f32, 2>> = Tensor::Cpu(
            fusor_cpu::Tensor::from_slice([3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );

        // Select column 0: [1, 3, 5]
        let indexed = tensor.i((.., 0));
        let slice = indexed.as_slice().await.unwrap();

        assert_eq!(slice[[0]], 1.0);
        assert_eq!(slice[[1]], 3.0);
        assert_eq!(slice[[2]], 5.0);
    }

    #[tokio::test]
    async fn test_index_3d() {
        // Create a 3D tensor [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        let tensor: Tensor<3, f32, ConcreteTensor<f32, 3>> = Tensor::Cpu(
            fusor_cpu::Tensor::from_slice([2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        );

        // Select along middle dimension: tensor[:, 0, :] -> [[1, 2], [5, 6]]
        let indexed = tensor.i((.., 0, ..));
        let slice = indexed.as_slice().await.unwrap();

        assert_eq!(slice[[0, 0]], 1.0);
        assert_eq!(slice[[0, 1]], 2.0);
        assert_eq!(slice[[1, 0]], 5.0);
        assert_eq!(slice[[1, 1]], 6.0);
    }

    #[tokio::test]
    async fn test_index_4d() {
        // Create a 4D tensor
        let tensor: Tensor<4, f32, ConcreteTensor<f32, 4>> = Tensor::Cpu(
            fusor_cpu::Tensor::from_slice([1, 2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        );

        // Select tensor[0, :, :, :] -> 3D tensor
        let indexed = tensor.i((0, .., .., ..));
        let slice = indexed.as_slice().await.unwrap();

        assert_eq!(slice[[0, 0, 0]], 1.0);
        assert_eq!(slice[[0, 0, 1]], 2.0);
        assert_eq!(slice[[1, 1, 1]], 8.0);
    }
}
