//! Axis reduction operations that work on both CPU and GPU backends.

use crate::cpu::{
    LastRank as CpuLastRank, MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp, TensorBacking,
};
use crate::gpu::{DataType, FloatDataType, LastRank as GpuLastRank};
use crate::{ConcreteTensor, DivOp, FloatOps, SimdBinaryOp, SimdElement, Tensor};

/// Emit a rank-reducing axis reduction method that dispatches CPU/GPU.
///
/// `$method` is the public name (e.g. `sum`).
/// `$op` is the SIMD reduce op marker (e.g. `SumOp`).
/// `$cpu_method` is the CPU tensor method (e.g. `sum_axis`).
/// `$gpu_method` is the GPU tensor method (e.g. `sum`).
macro_rules! axis_reduce {
    ($(#[$meta:meta])* $method:ident, $op:ident, $cpu_method:ident, $gpu_method:ident) => {
        $(#[$meta])*
        pub fn $method<const OUT_RANK: usize>(
            &self,
            axis: usize,
        ) -> Tensor<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
        where
            ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
            crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
            $op: SimdReduceOp<D>,
        {
            self.dispatch_ref(
                |t| t.as_ref().$cpu_method::<OUT_RANK>(axis),
                |t| t.$gpu_method(axis),
            )
        }
    };
}

/// Emit a keepdim variant that calls the rank-reducing form then reshapes.
///
/// `$method` is the public name (e.g. `sum_keepdim`).
/// `$base` is the rank-reducing method to call (e.g. `sum`).
macro_rules! axis_reduce_keepdim {
    ($(#[$meta:meta])* $method:ident, $base:ident, $op:ident) => {
        $(#[$meta])*
        pub fn $method<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
        where
            ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
            crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
            $op: SimdReduceOp<D>,
        {
            let mut kept_shape = self.shape();
            kept_shape[axis] = 1;
            self.$base::<OUT_RANK>(axis).reshape(kept_shape).to_concrete()
        }
    };
}

impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
    B: TensorBacking<R, Elem = D>,
{
    axis_reduce!(
        /// Sum along a specific axis, reducing the tensor rank by 1.
        ///
        /// # Arguments
        /// * `axis` - The axis to reduce along (0 to R-1)
        ///
        /// # Type Parameters
        /// - `OUT_RANK`: The output tensor rank (must be R - 1)
        sum, SumOp, sum_axis, sum
    );

    axis_reduce!(
        /// Maximum along a specific axis, reducing the tensor rank by 1.
        max, MaxOp, max_axis, max
    );

    axis_reduce!(
        /// Minimum along a specific axis, reducing the tensor rank by 1.
        min, MinOp, min_axis, min
    );

    axis_reduce!(
        /// Product along a specific axis, reducing the tensor rank by 1.
        product, ProdOp, prod_axis, product
    );

    axis_reduce_keepdim!(
        /// Product along a specific axis, keeping the reduced dimension with size 1.
        product_keepdim, product, ProdOp
    );

    axis_reduce_keepdim!(
        /// Sum along a specific axis, keeping the reduced dimension with size 1.
        sum_keepdim, sum, SumOp
    );

    axis_reduce_keepdim!(
        /// Max along a specific axis, keeping the reduced dimension with size 1.
        max_keepdim, max, MaxOp
    );

    axis_reduce_keepdim!(
        /// Min along a specific axis, keeping the reduced dimension with size 1.
        min_keepdim, min, MinOp
    );

    /// Mean along a specific axis, reducing the tensor rank by 1.
    pub fn mean<const OUT_RANK: usize>(
        &self,
        axis: usize,
    ) -> Tensor<OUT_RANK, D, ConcreteTensor<D, OUT_RANK>>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
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
