use fusor_types::{SlidingWindow, StrideSpec};

use super::*;

impl<const R: usize, T: AutogradElement> Tensor<R, T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
{
    pub(super) fn sum_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        let input_shape = self.shape();
        let value = self.value.sum_keepdim::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R, T>(&*gradient, "sum_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).into_concrete()),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub(super) fn sum_any<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK, T>
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        let input_shape = self.shape();
        let value = self.value.sum::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let mut keepdim_shape = input_shape;
        keepdim_shape[axis] = 1;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT_RANK, T>(&*gradient, "sum")?;
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
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        let input = self.value.clone();
        let value = input.max_keepdim::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R, T>(&*gradient, "max_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(reduction_extrema_keepdim_grad::<R, OUT_RANK, T>(
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
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        let input = self.value.clone();
        let value = input.min_keepdim::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R, T>(&*gradient, "min_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(reduction_extrema_keepdim_grad::<R, OUT_RANK, T>(
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
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.sum_keepdim_any::<OUT_RANK>(axis)
            .div_scalar(self.shape()[axis] as f32)
    }

    fn product_keepdim_any<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::ProdOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    {
        let input = self.value.clone();
        let input_shape = self.shape();
        let value = input.product_keepdim::<OUT_RANK>(axis).into_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R, T>(&*gradient, "product_keepdim")?;
            let upstream = gradient.broadcast_as(input_shape).into_concrete();
            let zeros = RawTensor::zeros(&input.device(), input_shape);
            let ones = RawTensor::splat(&input.device(), T::from_f32(1.0), input_shape);
            let zero_mask = input.eq(T::from_f32(0.0)).into_concrete();
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
            let gradient = ((no_zero_grad
                * zero_count_broadcast.eq(T::from_f32(0.0)).into_concrete())
            .into_concrete()
                + (single_zero_grad * zero_count_broadcast.eq(T::from_f32(1.0)).into_concrete())
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
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        let mean = self.mean_keepdim_any::<OUT_RANK>(axis);
        let centered = self.sub(&mean.broadcast_as(self.shape()));
        centered.sqr().mean_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn pool<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<crate::composite::pool::PoolSize>; DIFF],
        with: impl Fn(&Tensor<O, T>, usize) -> Self + Copy,
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LargerRank<R2, DIFF, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LargerRank<DIFF, R2, T>,
        crate::ConcreteTensor<T, R2>: crate::cpu::LargerRank<R3, 1, T>,
        crate::gpu::Tensor<R2, T>: crate::gpu::LargerRank<1, R3, T>,
        crate::gpu::Tensor<R3, T>: crate::gpu::SmallerRank<DIFF, O, T>,
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

        let tiled: Tensor<R2, T> = self.restride(specs);
        let unsqueezed: Tensor<R3, T> = tiled.unsqueeze_dims::<1, R3>([R2]);
        let flattened: Tensor<O, T> = unsqueezed.flatten_last_n::<DIFF, O>();
        with(&flattened, O - 1)
    }

    pub fn pool_max<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<crate::composite::pool::PoolSize>; DIFF],
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LargerRank<R2, DIFF, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LargerRank<DIFF, R2, T>,
        crate::ConcreteTensor<T, R2>: crate::cpu::LargerRank<R3, 1, T>,
        crate::gpu::Tensor<R2, T>: crate::gpu::LargerRank<1, R3, T>,
        crate::gpu::Tensor<R3, T>: crate::gpu::SmallerRank<DIFF, O, T>,
        crate::ConcreteTensor<T, O>: crate::cpu::LastRank<R, T>,
        crate::gpu::Tensor<O, T>: crate::gpu::LastRank<R, T> + crate::gpu::SmallerRank<1, R, T>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.pool::<DIFF, R2, R3, O>(pools, |windowed, axis| windowed.max::<R>(axis))
    }

    pub fn pool_min<const DIFF: usize, const R2: usize, const R3: usize, const O: usize>(
        &self,
        pools: [impl Into<crate::composite::pool::PoolSize>; DIFF],
    ) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LargerRank<R2, DIFF, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LargerRank<DIFF, R2, T>,
        crate::ConcreteTensor<T, R2>: crate::cpu::LargerRank<R3, 1, T>,
        crate::gpu::Tensor<R2, T>: crate::gpu::LargerRank<1, R3, T>,
        crate::gpu::Tensor<R3, T>: crate::gpu::SmallerRank<DIFF, O, T>,
        crate::ConcreteTensor<T, O>: crate::cpu::LastRank<R, T>,
        crate::gpu::Tensor<O, T>: crate::gpu::LastRank<R, T> + crate::gpu::SmallerRank<1, R, T>,
        crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.pool::<DIFF, R2, R3, O>(pools, |windowed, axis| windowed.min::<R>(axis))
    }

    pub fn max_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.max_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn max<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK, T>
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>:
            crate::gpu::LastRank<OUT_RANK, T> + crate::gpu::SmallerRank<1, OUT_RANK, T>,
        crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.max_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn min_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.min_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn min<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK, T>
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>:
            crate::gpu::LastRank<OUT_RANK, T> + crate::gpu::SmallerRank<1, OUT_RANK, T>,
        crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.min_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn mean_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.mean_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn mean<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK, T>
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>:
            crate::gpu::LastRank<OUT_RANK, T> + crate::gpu::SmallerRank<1, OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.mean_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn product<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK, T>
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>:
            crate::gpu::LastRank<OUT_RANK, T> + crate::gpu::SmallerRank<1, OUT_RANK, T>,
        crate::cpu::ProdOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    {
        self.product_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn product_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::ProdOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
        crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    {
        self.product_keepdim_any::<OUT_RANK>(axis)
    }

    pub fn var<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<OUT_RANK, T>
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>:
            crate::gpu::LastRank<OUT_RANK, T> + crate::gpu::SmallerRank<1, OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.var_keepdim_any::<OUT_RANK>(axis)
            .squeeze_dims::<1, OUT_RANK>([axis])
    }

    pub fn var_keepdim<const OUT_RANK: usize>(&self, axis: usize) -> Self
    where
        crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
        crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    {
        self.var_keepdim_any::<OUT_RANK>(axis)
    }
}

impl<T: AutogradElement> Tensor<1, T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
{
    pub fn sum(&self) -> Tensor<0, T> {
        let input_shape = self.shape();
        let value = self.value.sum::<0>(0);
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<0, T>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).into_concrete()),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum_keepdim(&self, axis: usize) -> Tensor<1, T> {
        self.sum_keepdim_any::<0>(axis)
    }
}

macro_rules! sum_wrappers {
    ($($rank:literal => $out:literal),* $(,)?) => {$(
        impl<T: AutogradElement> Tensor<$rank, T>
where
            crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
            crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
            crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
            crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
        {
            pub fn sum(&self, axis: usize) -> Tensor<$out, T>
            where
                crate::ConcreteTensor<T, $rank>: crate::cpu::LastRank<$out, T>,
                crate::gpu::Tensor<$rank, T>: crate::gpu::LastRank<$out, T>,
            {
                self.sum_any::<$out>(axis)
            }

            pub fn sum_keepdim(&self, axis: usize) -> Tensor<$rank, T>
            where
                crate::ConcreteTensor<T, $rank>: crate::cpu::LastRank<$out, T>,
                crate::gpu::Tensor<$rank, T>: crate::gpu::LastRank<$out, T>,
            {
                self.sum_keepdim_any::<$out>(axis)
            }
        }
    )*};
}

sum_wrappers!(2 => 1, 3 => 2, 4 => 3, 5 => 4, 6 => 5, 7 => 6, 8 => 7, 9 => 8, 10 => 9);

fn reduction_extrema_keepdim_grad<const R: usize, const OUT_RANK: usize, T: AutogradElement>(
    input: RawTensor<R, T>,
    axis: usize,
    gradient: RawTensor<R, T>,
    is_max: bool,
) -> RawTensor<R, T>
where
    crate::ConcreteTensor<T, R>: crate::cpu::LastRank<OUT_RANK, T>,
    crate::gpu::Tensor<R, T>: crate::gpu::LastRank<OUT_RANK, T>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
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
        .eq(T::from_f32(0.0))
        .into_concrete();
    let tie_count = mask
        .sum_keepdim::<OUT_RANK>(axis)
        .broadcast_as(input_shape)
        .into_concrete();
    ((mask * gradient.broadcast_as(input_shape)).into_concrete() / tie_count).into_concrete()
}
