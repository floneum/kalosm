//! Axis reduction operations that work on both CPU and GPU backends.

use crate::{AddOp, ConcreteTensor, DivOp, FloatOps, SimdBinaryOp, SimdElement, Tensor};
use fusor_core::{
    DataType, FloatDataType, LastRank as GpuLastRank, NextRankInner as GpuNextRankInner,
};
use fusor_cpu::{
    LastRank as CpuLastRank, MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp, TensorBacking,
};

impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Sum along a specific axis, reducing the tensor rank by 1.
    ///
    /// # Arguments
    /// * `axis` - The axis to reduce along (0 to R-1)
    ///
    /// # Type Parameters
    /// - `OUT_RANK`: The output tensor rank (must be R - 1)
    pub fn sum<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> Tensor<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        SumOp: SimdReduceOp<D>,
    {
        self.dispatch_ref(|t| t.as_ref().sum_axis::<OUT_RANK>(axis), |t| t.sum(axis))
    }

    /// Maximum along a specific axis, reducing the tensor rank by 1.
    pub fn max<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> Tensor<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        MaxOp: SimdReduceOp<D>,
    {
        self.dispatch_ref(|t| t.as_ref().max_axis::<OUT_RANK>(axis), |t| t.max(axis))
    }

    /// Minimum along a specific axis, reducing the tensor rank by 1.
    pub fn min<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> Tensor<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        MinOp: SimdReduceOp<D>,
    {
        self.dispatch_ref(|t| t.as_ref().min_axis::<OUT_RANK>(axis), |t| t.min(axis))
    }

    /// Product along a specific axis, reducing the tensor rank by 1.
    pub fn product<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> Tensor<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        ProdOp: SimdReduceOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().prod_axis::<OUT_RANK>(axis),
            |t| t.product(axis),
        )
    }

    /// Product along a specific axis, keeping the reduced dimension with size 1.
    pub fn product_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        ProdOp: SimdReduceOp<D>,
    {
        match self {
            Tensor::Cpu(t) => {
                let reduced = t.as_ref().prod_axis::<OUT_RANK>(axis);
                let original_shape: [usize; R] = t.shape();
                Tensor::Cpu(unsqueeze_at_axis::<R, OUT_RANK, D>(
                    &reduced,
                    original_shape,
                    axis,
                ))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.product_keepdim(axis)),
        }
    }

    /// Sum along a specific axis, keeping the reduced dimension with size 1.
    pub fn sum_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
    {
        match self {
            Tensor::Cpu(t) => {
                let reduced = t.as_ref().sum_axis::<OUT_RANK>(axis);
                let original_shape: [usize; R] = t.shape();
                Tensor::Cpu(unsqueeze_at_axis::<R, OUT_RANK, D>(
                    &reduced,
                    original_shape,
                    axis,
                ))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.sum_keepdim(axis)),
        }
    }

    /// Max along a specific axis, keeping the reduced dimension with size 1.
    pub fn max_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        MaxOp: SimdReduceOp<D>,
    {
        match self {
            Tensor::Cpu(t) => {
                let reduced = t.as_ref().max_axis::<OUT_RANK>(axis);
                let original_shape: [usize; R] = t.shape();
                Tensor::Cpu(unsqueeze_at_axis::<R, OUT_RANK, D>(
                    &reduced,
                    original_shape,
                    axis,
                ))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.max_keepdim(axis)),
        }
    }

    /// Min along a specific axis, keeping the reduced dimension with size 1.
    pub fn min_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        MinOp: SimdReduceOp<D>,
    {
        match self {
            Tensor::Cpu(t) => {
                let reduced = t.as_ref().min_axis::<OUT_RANK>(axis);
                let original_shape: [usize; R] = t.shape();
                Tensor::Cpu(unsqueeze_at_axis::<R, OUT_RANK, D>(
                    &reduced,
                    original_shape,
                    axis,
                ))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.min_keepdim(axis)),
        }
    }

    /// Mean along a specific axis, reducing the tensor rank by 1.
    pub fn mean<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> Tensor<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Div<Output = D>,
        DivOp: SimdBinaryOp<D>,
    {
        let shape = self.shape();
        let axis_size = shape[axis];
        let sum = self.sum::<OUT_RANK>(axis);
        sum.div_scalar(D::from_f32(axis_size as f32))
    }

    /// Mean along a specific axis, keeping the dimension (with size 1).
    pub fn mean_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Add<Output = D> + std::ops::Div<Output = D>,
        AddOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
    {
        let shape = self.shape();
        let axis_size = shape[axis];
        let sum = self.sum_keepdim::<OUT_RANK>(axis);
        sum.div_scalar(D::from_f32(axis_size as f32))
    }

    /// Variance along a specific axis, reducing the tensor rank by 1.
    ///
    /// Uses the formula: var(x) = mean(x^2) - mean(x)^2
    pub fn var<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> Tensor<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D> + std::ops::Sub<Output = D> + std::ops::Div<Output = D>,
        crate::MulOp: SimdBinaryOp<D>,
        crate::SubOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
    {
        // var(x) = mean(x^2) - mean(x)^2
        let concrete = self.to_concrete();
        let mean_x = concrete.mean::<OUT_RANK>(axis);
        let mean_x_sq = mean_x.sqr();
        let x_sq = concrete.sqr();
        let mean_x2 = x_sq.mean::<OUT_RANK>(axis);
        // mean(x^2) - mean(x)^2
        (&mean_x2 - &mean_x_sq).to_concrete()
    }

    /// Variance along a specific axis, keeping the dimension (with size 1).
    pub fn var_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D>
            + std::ops::Sub<Output = D>
            + std::ops::Add<Output = D>
            + std::ops::Div<Output = D>,
        crate::MulOp: SimdBinaryOp<D>,
        crate::SubOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
    {
        // var(x) = mean(x^2) - mean(x)^2
        let concrete = self.to_concrete();
        let mean_x = concrete.mean_keepdim::<OUT_RANK>(axis);
        let mean_x_sq = mean_x.sqr();
        let x_sq = concrete.sqr();
        let mean_x2 = x_sq.mean_keepdim::<OUT_RANK>(axis);
        // mean(x^2) - mean(x)^2
        (&mean_x2 - &mean_x_sq).to_concrete()
    }
}

/// Helper function to unsqueeze a reduced tensor back to original rank with size 1 at the axis.
/// The reduced tensor has OUT_RANK dimensions (one less than original R).
/// The result has R dimensions with size 1 at the reduced axis (standard keepdim semantics).
fn unsqueeze_at_axis<const R: usize, const OUT_RANK: usize, D>(
    reduced: &fusor_cpu::Tensor<OUT_RANK, ConcreteTensor<D, OUT_RANK>>,
    original_shape: [usize; R],
    axis: usize,
) -> fusor_cpu::Tensor<R, ConcreteTensor<D, R>>
where
    D: SimdElement + Default,
    ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
{
    // Build keepdim shape: same as original but with size 1 at axis
    let mut keepdim_shape = original_shape;
    keepdim_shape[axis] = 1;

    // Collect all elements from the reduced tensor
    let reduced_shape = reduced.shape();
    let total_elements: usize = reduced_shape.iter().product();
    let reduced_concrete = reduced.to_concrete();
    let data: Vec<D> = (0..total_elements)
        .map(|i| {
            // Convert linear index to reduced shape indices
            let mut indices = [0usize; OUT_RANK];
            let mut remainder = i;
            for dim in (0..OUT_RANK).rev() {
                indices[dim] = remainder % reduced_shape[dim];
                remainder /= reduced_shape[dim];
            }
            reduced_concrete.get(indices)
        })
        .collect();
    fusor_cpu::Tensor::from_slice(keepdim_shape, &data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sum_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
        let result: Tensor<1, f32, _> = t.sum::<1>(0);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 5.0).abs() < 0.001);
        assert!((slice[[1]] - 7.0).abs() < 0.001);
        assert!((slice[[2]] - 9.0).abs() < 0.001);

        // Sum along axis 1: [1+2+3, 4+5+6] = [6, 15]
        let result: Tensor<1, f32, _> = t.sum::<1>(1);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 6.0).abs() < 0.001);
        assert!((slice[[1]] - 15.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_max_cpu() {
        let data = [1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Max along axis 0: [max(1,4), max(5,2), max(3,6)] = [4, 5, 6]
        let result: Tensor<1, f32, _> = t.max::<1>(0);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 4.0).abs() < 0.001);
        assert!((slice[[1]] - 5.0).abs() < 0.001);
        assert!((slice[[2]] - 6.0).abs() < 0.001);

        // Max along axis 1: [max(1,5,3), max(4,2,6)] = [5, 6]
        let result: Tensor<1, f32, _> = t.max::<1>(1);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 5.0).abs() < 0.001);
        assert!((slice[[1]] - 6.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_min_cpu() {
        let data = [1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Min along axis 0: [min(1,4), min(5,2), min(3,6)] = [1, 2, 3]
        let result: Tensor<1, f32, _> = t.min::<1>(0);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 1.0).abs() < 0.001);
        assert!((slice[[1]] - 2.0).abs() < 0.001);
        assert!((slice[[2]] - 3.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_sum_keepdim_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Sum along axis 1, keepdim -> shape [2, 1]
        // Row 0 sum: 1+2+3 = 6, Row 1 sum: 4+5+6 = 15
        let result = t.sum_keepdim::<1>(1);
        assert_eq!(result.shape(), [2, 1]); // Keepdim shape
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0, 0]] - 6.0).abs() < 0.001);
        assert!((slice[[1, 0]] - 15.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_mean_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Mean along axis 1: [(1+2+3)/3, (4+5+6)/3] = [2, 5]
        let result: Tensor<1, f32, _> = t.mean::<1>(1);
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0]] - 2.0).abs() < 0.001);
        assert!((slice[[1]] - 5.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_mean_keepdim_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Mean along axis 1, keepdim -> shape [2, 1]
        let result = t.mean_keepdim::<1>(1);
        assert_eq!(result.shape(), [2, 1]); // Keepdim shape
        let slice = result.as_slice().await.unwrap();
        assert!((slice[[0, 0]] - 2.0).abs() < 0.001);
        assert!((slice[[1, 0]] - 5.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_var_cpu() {
        // Test variance: var([1, 2, 3]) = mean([1, 4, 9]) - mean([1, 2, 3])^2
        //                              = 14/3 - 4 = 2/3 ≈ 0.6667
        let data = [1.0f32, 2.0, 3.0];
        let t: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &data));

        let result: Tensor<0, f32, _> = t.var::<0>(0);
        let slice = result.as_slice().await.unwrap();
        let expected = 2.0 / 3.0; // population variance
        assert!(
            (slice[[]] - expected).abs() < 0.001,
            "Expected {}, got {}",
            expected,
            slice[[]]
        );
    }

    #[tokio::test]
    async fn test_var_2d_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        // Variance along axis 1
        // Row 0: [1, 2, 3] -> mean=2, var = (1+4+9)/3 - 4 = 14/3 - 4 = 2/3
        // Row 1: [4, 5, 6] -> mean=5, var = (16+25+36)/3 - 25 = 77/3 - 25 = 2/3
        let result: Tensor<1, f32, _> = t.var::<1>(1);
        let slice = result.as_slice().await.unwrap();
        let expected = 2.0 / 3.0;
        assert!((slice[[0]] - expected).abs() < 0.001);
        assert!((slice[[1]] - expected).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_var_keepdim_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        let result = t.var_keepdim::<1>(1);
        assert_eq!(result.shape(), [2, 1]); // Keepdim shape
        let slice = result.as_slice().await.unwrap();
        let expected = 2.0 / 3.0;
        assert!((slice[[0, 0]] - expected).abs() < 0.001);
        assert!((slice[[1, 0]] - expected).abs() < 0.001);
    }
}
