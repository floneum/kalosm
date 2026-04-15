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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pool_1d_cpu() {
        // Test 1D pooling with batch and channel dimensions (3D input like fusor-core)
        // Input: (1, 1, 12) - batch=1, channels=1, length=12
        let input_data = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 3.0, 12.0, 3.0, 5.0, 39.0, 29.0, 1.0,
        ];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 12], &input_data));

        for (pool_size, stride) in [(2, 2), (3, 1), (4, 2)] {
            let output = input.pool_max([(pool_size, stride)]);

            let expected: Vec<f32> = input_data
                .windows(pool_size)
                .step_by(stride)
                .map(|window| window.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
                .collect();

            let output_data = output.as_slice().await.unwrap();
            assert_eq!(output_data.shape(), &[1, 1, expected.len()]);
            for i in 0..expected.len() {
                let val = output_data[[0, 0, i]];
                let expected_val = expected[i];
                assert!(
                    (val - expected_val).abs() < 1e-6,
                    "Mismatch at index {}: got {}, expected {}",
                    i,
                    val,
                    expected_val
                );
            }
        }
    }

    #[tokio::test]
    async fn test_pool_1d_multi_channel_cpu() {
        let pool_size = 3;
        let stride = 2;

        // Input: (2, 4, 16) - batch=2, channels=4, length=16
        let mut input_data = vec![];
        for b in 0..2 {
            for c in 0..4 {
                for i in 0..16 {
                    let val = ((b * 64 + c * 16 + i) % 13) as f32;
                    input_data.push(val);
                }
            }
        }

        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 4, 16], &input_data));

        let output = input.pool_max([(pool_size, stride)]);
        let output_data = output.as_slice().await.unwrap();

        let fusor_shape = output_data.shape();
        assert_eq!(fusor_shape[0], 2);
        assert_eq!(fusor_shape[1], 4);

        // Manually compute expected values
        for b in 0..2 {
            for c in 0..4 {
                // Extract this channel's data
                let channel: Vec<f32> = (0..16)
                    .map(|i| ((b * 64 + c * 16 + i) % 13) as f32)
                    .collect();
                let expected: Vec<f32> = channel
                    .windows(pool_size)
                    .step_by(stride)
                    .map(|window| window.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
                    .collect();

                for i in 0..fusor_shape[2] {
                    let fusor_val = output_data[[b, c, i]];
                    let expected_val = expected[i];
                    assert!(
                        (fusor_val - expected_val).abs() < 1e-5,
                        "Mismatch at [{}, {}, {}]: got {}, expected {}",
                        b,
                        c,
                        i,
                        fusor_val,
                        expected_val
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_pool_min_cpu() {
        // Input: (1, 1, 6) - batch=1, channels=1, length=6
        let input_data = [5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 6], &input_data));

        let output = input.pool_min([(2, 2)]);
        let output_data = output.as_slice().await.unwrap();

        // Pool windows: [5,2], [8,1], [9,3]
        // Min: 2, 1, 3
        assert_eq!(output_data[[0, 0, 0]], 2.0);
        assert_eq!(output_data[[0, 0, 1]], 1.0);
        assert_eq!(output_data[[0, 0, 2]], 3.0);
    }
}
