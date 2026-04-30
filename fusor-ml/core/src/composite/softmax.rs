use crate::{DataType, Dim, LastRank, Tensor};

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn softmax<const R2: usize>(&self, axis: impl Dim<R>) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
    {
        let axis = axis.resolve();
        let mut kept_shape = *self.shape();
        kept_shape[axis] = 1;
        let max = self
            .max::<R2>(axis)
            .reshape(kept_shape)
            .broadcast_as(*self.shape());
        let shifted = self - &max;
        let exp = shifted.exp();
        let sum = exp
            .sum::<R2>(axis)
            .reshape(kept_shape)
            .broadcast_as(*self.shape());
        &exp / &sum
    }

    pub fn softmax_last_dim<const R2: usize>(&self) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
    {
        self.softmax(crate::D::Minus1)
    }
}
