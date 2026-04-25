use crate::{DataType, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Create a tensor filled with ones that has the same shape as this tensor
    pub fn ones_like(&self) -> Self {
        Self::splat(self.device(), D::one(), *self.shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ones_like() {
        use crate::Device;

        let device = Device::new().await.unwrap();

        let data = [[1., 2.], [3., 4.]];
        let tensor = Tensor::new(&device, &data);
        let ones = tensor.ones_like();

        assert_eq!(ones.shape(), tensor.shape());

        let ones_slice = ones.as_slice().await.unwrap();
        assert_eq!(ones_slice[[0, 0]], 1.);
        assert_eq!(ones_slice[[0, 1]], 1.);
        assert_eq!(ones_slice[[1, 0]], 1.);
        assert_eq!(ones_slice[[1, 1]], 1.);
    }
}
