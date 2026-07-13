use fusor_types::{SlidingWindow, StrideSpec};

use super::*;

impl<const R: usize> Tensor<R> {
    pub(super) fn sum_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        let input_shape = self.shape();
        let value = self.value.sum_keepdim::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "sum_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).into_concrete()),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub(super) fn sum_any<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        let input_shape = self.shape();
        let value = self.value.sum::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let mut keepdim_shape = input_shape;
        keepdim_shape[axis] = 1;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT_RANK>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(
                    gradient
                        .reshape(keepdim_shape)
                        .into_concrete()
                        .broadcast_as(input_shape)
                        .into_concrete(),
                ),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub(super) fn max_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        let input = self.value.clone();
        let value = input.max_keepdim::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "max_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(reduction_extrema_keepdim_grad::<R, OUT_RANK>(
                    input.clone(),
                    axis,
                    gradient,
                    true,
                )),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn min_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::MinOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        let input = self.value.clone();
        let value = input.min_keepdim::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "min_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(reduction_extrema_keepdim_grad::<R, OUT_RANK>(
                    input.clone(),
                    axis,
                    gradient,
                    false,
                )),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub(super) fn mean_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.sum_keepdim_any::<OUT_RANK>(axis)
            .div_scalar(self.shape()[axis] as f32)
    }

    fn product_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::ProdOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::EqOp: crate::cpu::SimdBinaryOp<f32>,
    {
        let input = self.value.clone();
        let input_shape = self.shape();
        let value = input.product_keepdim::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "product_keepdim")?;
            let upstream = gradient.broadcast_as(input_shape).into_concrete();
            let zeros = RawTensor::zeros(&input.device(), input_shape);
            let ones = RawTensor::splat(&input.device(), 1.0, input_shape);
            let zero_mask = input.eq(0.0).into_concrete();
            let safe_input = zero_mask.where_cond(&ones, &input).into_concrete();
            let zero_count = zero_mask.sum_keepdim::<OUT_RANK>(axis).into_concrete();
            let zero_count_broadcast = zero_count.broadcast_as(input_shape).into_concrete();
            let product_non_zero = safe_input
                .product_keepdim::<OUT_RANK>(axis)
                .broadcast_as(input_shape)
                .into_concrete();
            let no_zero_grad = (upstream.clone()
                * (product_non_zero.clone() / safe_input).into_concrete())
            .into_concrete();
            let single_zero_grad = zero_mask
                .where_cond(&(upstream * product_non_zero).into_concrete(), &zeros)
                .into_concrete();
            let gradient = ((no_zero_grad * zero_count_broadcast.eq(0.0).into_concrete())
                .into_concrete()
                + (single_zero_grad * zero_count_broadcast.eq(1.0).into_concrete())
                    .into_concrete())
            .into_concrete();
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn var_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        let mean = self.mean_keepdim_any::<OUT_RANK>(axis);
        let centered = self.sub(&mean.broadcast_as(self.shape()));
        centered.sqr().mean_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn pool<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<crate::composite::pool::PoolSize>; DIFF],
        with: impl Fn(&Tensor<O>, usize) -> Self + Copy,
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LargerRank<R2, DIFF, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LargerRank<DIFF, R2, f32>,
        crate::ConcreteTensor<f32, R2>: crate::cpu::LargerRank<R3, 1, f32>,
        crate::gpu::Tensor<R2, f32>: crate::gpu::LargerRank<1, R3, f32>,
        crate::gpu::Tensor<R3, f32>: crate::gpu::SmallerRank<DIFF, O, f32>,
    {
        let pools: [crate::composite::pool::PoolSize; DIFF] = pools.map(|pool| pool.into());
        let axis_start = R - DIFF;
        let windows: [SlidingWindow; DIFF] = std::array::from_fn(|i| {
            let pool = pools[i];
            SlidingWindow::new(axis_start + i, pool.size, pool.stride)
        });
        let shape = self.shape();
        let mut sorted_windows = windows;
        sorted_windows.sort_by_key(|window| window.axis);
        let specs: [StrideSpec; R2] = std::array::from_fn(|out_i| {
            if out_i < R {
                if let Some(window) = sorted_windows.iter().find(|window| window.axis == out_i) {
                    let positions = (shape[out_i] - window.window_size) / window.step + 1;
                    StrideSpec::dim_with(out_i, positions, window.step)
                } else {
                    StrideSpec::dim(out_i, shape[out_i])
                }
            } else {
                let window = &sorted_windows[out_i - R];
                StrideSpec::dim(window.axis, window.window_size)
            }
        });

        let tiled: Tensor<R2> = self.restride(specs);
        let unsqueezed: Tensor<R3> = tiled.unsqueeze_dims::<1, R3>([R2]);
        let flattened: Tensor<O> = unsqueezed.flatten_last_n::<DIFF, O>();
        with(&flattened, O - 1)
    }

    pub fn pool_max<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<crate::composite::pool::PoolSize>; DIFF],
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LargerRank<R2, DIFF, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LargerRank<DIFF, R2, f32>,
        crate::ConcreteTensor<f32, R2>: crate::cpu::LargerRank<R3, 1, f32>,
        crate::gpu::Tensor<R2, f32>: crate::gpu::LargerRank<1, R3, f32>,
        crate::gpu::Tensor<R3, f32>: crate::gpu::SmallerRank<DIFF, O, f32>,
        crate::ConcreteTensor<f32, O>: crate::cpu::LastRank<R, f32>,
        crate::gpu::Tensor<O, f32>:
            crate::gpu::LastRank<R, f32> + crate::gpu::SmallerRank<1, R, f32>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.pool::<DIFF, R2, R3, O>(pools, |windowed, axis| windowed.max::<R>(axis))
    }

    pub fn pool_min<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<crate::composite::pool::PoolSize>; DIFF],
    ) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LargerRank<R2, DIFF, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LargerRank<DIFF, R2, f32>,
        crate::ConcreteTensor<f32, R2>: crate::cpu::LargerRank<R3, 1, f32>,
        crate::gpu::Tensor<R2, f32>: crate::gpu::LargerRank<1, R3, f32>,
        crate::gpu::Tensor<R3, f32>: crate::gpu::SmallerRank<DIFF, O, f32>,
        crate::ConcreteTensor<f32, O>: crate::cpu::LastRank<R, f32>,
        crate::gpu::Tensor<O, f32>:
            crate::gpu::LastRank<R, f32> + crate::gpu::SmallerRank<1, R, f32>,
        crate::cpu::MinOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.pool::<DIFF, R2, R3, O>(pools, |windowed, axis| windowed.min::<R>(axis))
    }

    pub fn max_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.max_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn max<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>:
            crate::gpu::LastRank<OUT_RANK, f32> + crate::gpu::SmallerRank<1, OUT_RANK, f32>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.max_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn min_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::MinOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.min_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn min<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>:
            crate::gpu::LastRank<OUT_RANK, f32> + crate::gpu::SmallerRank<1, OUT_RANK, f32>,
        crate::cpu::MinOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.min_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn mean_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.mean_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn mean<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>:
            crate::gpu::LastRank<OUT_RANK, f32> + crate::gpu::SmallerRank<1, OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.mean_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn product<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>:
            crate::gpu::LastRank<OUT_RANK, f32> + crate::gpu::SmallerRank<1, OUT_RANK, f32>,
        crate::cpu::ProdOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::EqOp: crate::cpu::SimdBinaryOp<f32>,
    {
        self.product_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn product_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::ProdOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
        crate::cpu::EqOp: crate::cpu::SimdBinaryOp<f32>,
    {
        self.product_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn var<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK>
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>:
            crate::gpu::LastRank<OUT_RANK, f32> + crate::gpu::SmallerRank<1, OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.var_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn var_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
    {
        self.var_keepdim_any::<OUT_RANK>(axis)
    }
}

impl Tensor<1> {
    pub fn sum(&self) -> Tensor<0> {
        let input_shape = self.shape();
        let value = self.value.sum::<0>(0);
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<0>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).into_concrete()),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum_keepdim(&self, axis: usize) -> Tensor<1> {
        self.sum_keepdim_any::<0>(axis)
    }
}

macro_rules! sum_wrappers {
    ($($rank:literal => $out:literal),* $(,)?) => {$(
        impl Tensor<$rank> {
            pub fn sum(&self, axis: usize) -> Tensor<$out> {
                self.sum_any::<$out>(axis)
            }

            pub fn sum_keepdim(&self, axis: usize) -> Tensor<$rank> {
                self.sum_keepdim_any::<$out>(axis)
            }
        }
    )*};
}

sum_wrappers!(2 => 1, 3 => 2, 4 => 3, 5 => 4, 6 => 5, 7 => 6, 8 => 7, 9 => 8, 10 => 9);

fn reduction_extrema_keepdim_grad<const R: usize, const OUT_RANK: usize>(
    input: RawTensor<R, f32>,
    axis: usize,
    gradient: RawTensor<R, f32>,
    is_max: bool,
) -> RawTensor<R, f32>
where
    crate::ConcreteTensor<f32, R>: crate::cpu::LastRank<OUT_RANK, f32>,
    crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<f32>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<f32>,
{
    let input_shape = input.shape();
    let extrema = if is_max {
        input.max_keepdim::<OUT_RANK>(axis)
    } else {
        input.min_keepdim::<OUT_RANK>(axis)
    }
    .into_concrete();
    let extrema_broadcast = extrema.broadcast_as(input_shape).into_concrete();
    let mask = (input - extrema_broadcast)
        .into_concrete()
        .eq(0.0)
        .into_concrete();
    let tie_count = mask
        .sum_keepdim::<OUT_RANK>(axis)
        .broadcast_as(input_shape)
        .into_concrete();
    ((mask * gradient.broadcast_as(input_shape)).into_concrete() / tie_count).into_concrete()
}
