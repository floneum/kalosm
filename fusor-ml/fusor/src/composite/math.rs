//! Math operations that work on both CPU and GPU backends.

use crate::cpu::Mul;
use crate::gpu::{DataType, FloatDataType};
use crate::{ConcreteTensor, FloatOps, MulOp, ResolvedTensor, SimdBinaryOp, SimdElement, Tensor};

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Square each element: sqr(x) = x * x
    pub fn sqr(&self) -> Tensor<R, D, Mul<D, R, &ConcreteTensor<D, R>, &ConcreteTensor<D, R>>>
    where
        D: std::ops::Mul<Output = D>,
        MulOp: SimdBinaryOp<D>,
    {
        self * self
    }
}

impl<const R: usize, D> Tensor<R, D, ConcreteTensor<D, R>>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Element-wise power: pow(self, other) computes self^other for each element.
    pub fn pow(&self, other: &Self) -> Self {
        self.dispatch_pair(
            other,
            |a, b| {
                // Use element-wise powf via iterating
                let shape: [usize; R] = a.shape();
                let a_data = ResolvedTensor::data(a.inner());
                let b_data = ResolvedTensor::data(b.inner());
                let result: Vec<D> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(x, y)| x.powf(*y))
                    .collect();
                crate::cpu::TypedTensor::new(crate::cpu::ConcreteTensor::from_slice(shape, &result))
            },
            |a, b| a.pow(b),
        )
    }

    /// Resize tensor to new shape with padding/truncation.
    pub fn resize(&self, new_shape: [usize; R]) -> Self {
        match self {
            Tensor::Cpu(t) => {
                // CPU resize: create new tensor and copy elements
                let old_shape = self.shape();
                let src_data = ResolvedTensor::data(t.inner());
                let mut result = vec![D::default(); new_shape.iter().product()];

                // Calculate how many elements to copy per dimension
                let copy_shape: [usize; R] =
                    std::array::from_fn(|i| old_shape[i].min(new_shape[i]));

                // Copy elements using nested iteration
                struct CopyContext<'a, D, const R: usize> {
                    src: &'a [D],
                    old_shape: &'a [usize; R],
                    new_shape: &'a [usize; R],
                    copy_shape: &'a [usize; R],
                }

                fn copy_recursive<D: Copy, const R: usize>(
                    ctx: &CopyContext<'_, D, R>,
                    dst: &mut [D],
                    dim: usize,
                    src_offset: usize,
                    dst_offset: usize,
                ) {
                    if dim == R {
                        dst[dst_offset] = ctx.src[src_offset];
                    } else {
                        let old_stride: usize = ctx.old_shape[dim + 1..].iter().product();
                        let new_stride: usize = ctx.new_shape[dim + 1..].iter().product();
                        for i in 0..ctx.copy_shape[dim] {
                            copy_recursive(
                                ctx,
                                dst,
                                dim + 1,
                                src_offset + i * old_stride,
                                dst_offset + i * new_stride,
                            );
                        }
                    }
                }

                let ctx = CopyContext {
                    src: src_data.as_ref(),
                    old_shape: &old_shape,
                    new_shape: &new_shape,
                    copy_shape: &copy_shape,
                };
                copy_recursive(&ctx, &mut result, 0, 0, 0);

                Tensor::Cpu(crate::cpu::TypedTensor::new(
                    crate::cpu::ConcreteTensor::from_slice(new_shape, &result),
                ))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.resize(new_shape)),
        }
    }
}
