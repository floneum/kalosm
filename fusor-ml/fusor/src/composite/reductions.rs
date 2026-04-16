//! Axis reduction operations that work on both CPU and GPU backends.

use crate::{ConcreteTensor, DivOp, FloatOps, SimdBinaryOp, SimdElement, Tensor};
use fusor_core::{DataType, FloatDataType, LastRank as GpuLastRank};
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
        ProdOp: SimdReduceOp<D>,
    {
        let mut kept_shape = self.shape();
        kept_shape[axis] = 1;
        self.product::<OUT_RANK>(axis)
            .reshape(kept_shape)
            .to_concrete()
    }

    /// Sum along a specific axis, keeping the reduced dimension with size 1.
    pub fn sum_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        SumOp: SimdReduceOp<D>,
    {
        let mut kept_shape = self.shape();
        kept_shape[axis] = 1;
        self.sum::<OUT_RANK>(axis).reshape(kept_shape).to_concrete()
    }

    /// Max along a specific axis, keeping the reduced dimension with size 1.
    pub fn max_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        MaxOp: SimdReduceOp<D>,
    {
        let mut kept_shape = self.shape();
        kept_shape[axis] = 1;
        self.max::<OUT_RANK>(axis).reshape(kept_shape).to_concrete()
    }

    /// Min along a specific axis, keeping the reduced dimension with size 1.
    pub fn min_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        MinOp: SimdReduceOp<D>,
    {
        let mut kept_shape = self.shape();
        kept_shape[axis] = 1;
        self.min::<OUT_RANK>(axis).reshape(kept_shape).to_concrete()
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
        SumOp: SimdReduceOp<D>,
        D: std::ops::Div<Output = D>,
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
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D> + std::ops::Sub<Output = D> + std::ops::Div<Output = D>,
        crate::MulOp: SimdBinaryOp<D>,
        crate::SubOp: SimdBinaryOp<D>,
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
