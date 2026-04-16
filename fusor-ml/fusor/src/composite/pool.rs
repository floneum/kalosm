//! Pooling operations that work on both CPU and GPU backends.

use crate::{ConcreteTensor, FloatOps, SimdElement, Tensor};
use fusor_core::{DataType, FloatDataType};
use fusor_types::SlidingWindow;

/// Configuration for pooling operations
#[derive(Clone, Copy, Debug)]
pub struct PoolSize {
    pub size: usize,
    pub stride: usize,
}

impl From<usize> for PoolSize {
    fn from(size: usize) -> Self {
        Self { size, stride: size }
    }
}

impl From<(usize, usize)> for PoolSize {
    fn from((size, stride): (usize, usize)) -> Self {
        Self { size, stride }
    }
}

impl From<[usize; 2]> for PoolSize {
    fn from([size, stride]: [usize; 2]) -> Self {
        Self { size, stride }
    }
}

impl PoolSize {
    pub fn new(size: usize, stride: usize) -> Self {
        Self { size, stride }
    }
}

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Pooling operation that creates sliding windows and reduces them.
    ///
    /// # Type Parameters
    /// * `DIFF` - Number of spatial dimensions to pool over
    /// * `R2` - Intermediate rank after sliding window
    /// * `R3` - Intermediate rank after unsqueeze
    /// * `O` - Rank after flattening
    ///
    /// # Arguments
    /// * `pools` - Array of pool sizes for each spatial dimension
    /// * `with` - Reduction function to apply (e.g., max, min, mean)
    pub fn pool<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<PoolSize>; DIFF],
        with: fn(&Tensor<O, D, ConcreteTensor<D, O>>, usize) -> Self,
    ) -> Self
    where
        ConcreteTensor<D, R>: fusor_cpu::LargerRank<R2, DIFF, D>,
        fusor_core::Tensor<R, D>: fusor_core::LargerRank<DIFF, R2, D>,
        ConcreteTensor<D, R2>: fusor_cpu::NextRank<R3, D>,
        fusor_core::Tensor<R2, D>: fusor_core::NextRank<R3, D>,
        fusor_core::Tensor<R3, D>: fusor_core::SmallerRank<DIFF, O, D>,
        ConcreteTensor<D, O>: fusor_cpu::LastRank<R, D>,
        fusor_core::Tensor<O, D>: fusor_core::LastRank<R, D>,
    {
        let pools: [PoolSize; DIFF] = pools.map(|p| p.into());

        let axis_start = R - DIFF;
        let windows: [SlidingWindow; DIFF] = std::array::from_fn(|i| {
            let window = pools[i].size;
            let stride = pools[i].stride;
            SlidingWindow::new(axis_start + i, window, stride)
        });

        let tiled: Tensor<R2, D, _> = self.sliding_window_view(windows);

        let unsqueezed: Tensor<R3, D, _> = tiled.unsqueeze(R2);
        let flattened: Tensor<O, D, _> = unsqueezed.flatten_last_n::<DIFF, O>();

        with(&flattened, O - 1)
    }

    /// Max pooling operation.
    ///
    /// Applies sliding window and takes the maximum value in each window.
    pub fn pool_max<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<PoolSize>; DIFF],
    ) -> Self
    where
        ConcreteTensor<D, R>: fusor_cpu::LargerRank<R2, DIFF, D>,
        fusor_core::Tensor<R, D>: fusor_core::LargerRank<DIFF, R2, D>,
        ConcreteTensor<D, R2>: fusor_cpu::NextRank<R3, D>,
        fusor_core::Tensor<R2, D>: fusor_core::NextRank<R3, D>,
        fusor_core::Tensor<R3, D>: fusor_core::SmallerRank<DIFF, O, D>,
        ConcreteTensor<D, O>: fusor_cpu::LastRank<R, D>,
        fusor_core::Tensor<O, D>: fusor_core::LastRank<R, D>,
        fusor_cpu::MaxOp: fusor_cpu::SimdReduceOp<D>,
    {
        self.pool(pools, Tensor::max)
    }

    /// Min pooling operation.
    ///
    /// Applies sliding window and takes the minimum value in each window.
    pub fn pool_min<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<PoolSize>; DIFF],
    ) -> Self
    where
        ConcreteTensor<D, R>: fusor_cpu::LargerRank<R2, DIFF, D>,
        fusor_core::Tensor<R, D>: fusor_core::LargerRank<DIFF, R2, D>,
        ConcreteTensor<D, R2>: fusor_cpu::NextRank<R3, D>,
        fusor_core::Tensor<R2, D>: fusor_core::NextRank<R3, D>,
        fusor_core::Tensor<R3, D>: fusor_core::SmallerRank<DIFF, O, D>,
        ConcreteTensor<D, O>: fusor_cpu::LastRank<R, D>,
        fusor_core::Tensor<O, D>: fusor_core::LastRank<R, D>,
        fusor_cpu::MinOp: fusor_cpu::SimdReduceOp<D>,
    {
        self.pool(pools, Tensor::min)
    }
}
