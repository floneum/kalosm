//! Comparison operations that work on both CPU and GPU backends.
//!
//! These operations return tensors with 1.0 for true and 0.0 for false.

use crate::{SimdElement, Tensor};
use fusor_core::DataType;
use fusor_cpu::{EqOp, GtOp, GteOp, LtOp, LteOp, SimdBinaryOp};

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Element-wise equality comparison between two tensors.
    ///
    /// Returns 1.0 where elements are equal, 0.0 otherwise.
    /// Note: GPU comparison is only available for CPU tensors at this time.
    pub fn eq_tensor(&self, rhs: &Self) -> Self
    where
        EqOp: SimdBinaryOp<D>,
    {
        self.dispatch_cpu_only_pair(rhs, |a, b| a.as_ref().eq(b.as_ref()).to_concrete())
    }

    /// Element-wise less-than comparison between two tensors.
    ///
    /// Returns 1.0 where self < rhs, 0.0 otherwise.
    /// Note: GPU comparison is only available for CPU tensors at this time.
    pub fn lt_tensor(&self, rhs: &Self) -> Self
    where
        LtOp: SimdBinaryOp<D>,
    {
        self.dispatch_cpu_only_pair(rhs, |a, b| a.as_ref().lt(b.as_ref()).to_concrete())
    }

    /// Element-wise less-than-or-equal comparison between two tensors.
    ///
    /// Returns 1.0 where self <= rhs, 0.0 otherwise.
    /// Note: GPU comparison is only available for CPU tensors at this time.
    pub fn lte_tensor(&self, rhs: &Self) -> Self
    where
        LteOp: SimdBinaryOp<D>,
    {
        self.dispatch_cpu_only_pair(rhs, |a, b| a.as_ref().lte(b.as_ref()).to_concrete())
    }

    /// Element-wise greater-than comparison between two tensors.
    ///
    /// Returns 1.0 where self > rhs, 0.0 otherwise.
    /// Note: GPU comparison is only available for CPU tensors at this time.
    pub fn gt_tensor(&self, rhs: &Self) -> Self
    where
        GtOp: SimdBinaryOp<D>,
    {
        self.dispatch_cpu_only_pair(rhs, |a, b| a.as_ref().gt(b.as_ref()).to_concrete())
    }

    /// Element-wise greater-than-or-equal comparison between two tensors.
    ///
    /// Returns 1.0 where self >= rhs, 0.0 otherwise.
    /// Note: GPU comparison is only available for CPU tensors at this time.
    pub fn gte_tensor(&self, rhs: &Self) -> Self
    where
        GteOp: SimdBinaryOp<D>,
    {
        self.dispatch_cpu_only_pair(rhs, |a, b| a.as_ref().gte(b.as_ref()).to_concrete())
    }

    /// Element-wise equality comparison with a scalar.
    ///
    /// Returns 1.0 where elements equal the scalar, 0.0 otherwise.
    pub fn eq_scalar(&self, scalar: D) -> Self
    where
        EqOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().eq_scalar(scalar).to_concrete(),
            |t| t.eq(scalar),
        )
    }

    /// Element-wise less-than comparison with a scalar.
    ///
    /// Returns 1.0 where self < scalar, 0.0 otherwise.
    pub fn lt_scalar(&self, scalar: D) -> Self
    where
        LtOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().lt_scalar(scalar).to_concrete(),
            |t| t.lt(scalar),
        )
    }

    /// Element-wise less-than-or-equal comparison with a scalar.
    ///
    /// Returns 1.0 where self <= scalar, 0.0 otherwise.
    pub fn lte_scalar(&self, scalar: D) -> Self
    where
        LteOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().lte_scalar(scalar).to_concrete(),
            |t| t.lte(scalar),
        )
    }

    /// Element-wise greater-than comparison with a scalar.
    ///
    /// Returns 1.0 where self > scalar, 0.0 otherwise.
    pub fn gt_scalar(&self, scalar: D) -> Self
    where
        GtOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().gt_scalar(scalar).to_concrete(),
            |t| t.mt(scalar),
        )
    }

    /// Element-wise greater-than-or-equal comparison with a scalar.
    ///
    /// Returns 1.0 where self >= scalar, 0.0 otherwise.
    pub fn gte_scalar(&self, scalar: D) -> Self
    where
        GteOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().gte_scalar(scalar).to_concrete(),
            |t| t.mte(scalar),
        )
    }

    /// Element-wise equality comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where elements equal the scalar, 0.0 otherwise.
    /// This is an alias for `eq_scalar` to match fusor-core's API.
    pub fn eq(&self, rhs: D) -> Self
    where
        EqOp: SimdBinaryOp<D>,
    {
        self.eq_scalar(rhs)
    }

    /// Element-wise less-than comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where self < scalar, 0.0 otherwise.
    /// This is an alias for `lt_scalar` to match fusor-core's API.
    pub fn lt(&self, rhs: D) -> Self
    where
        LtOp: SimdBinaryOp<D>,
    {
        self.lt_scalar(rhs)
    }

    /// Element-wise less-than-or-equal comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where self <= scalar, 0.0 otherwise.
    /// This is an alias for `lte_scalar` to match fusor-core's API.
    pub fn lte(&self, rhs: D) -> Self
    where
        LteOp: SimdBinaryOp<D>,
    {
        self.lte_scalar(rhs)
    }

    /// Element-wise greater-than comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where self > scalar, 0.0 otherwise.
    /// This is an alias for `gt_scalar` to match fusor-core's API.
    /// Named `mt` (more than) to match fusor-core.
    pub fn mt(&self, rhs: D) -> Self
    where
        GtOp: SimdBinaryOp<D>,
    {
        self.gt_scalar(rhs)
    }

    /// Element-wise greater-than-or-equal comparison with a scalar (fusor-core compatible API).
    ///
    /// Returns 1.0 where self >= scalar, 0.0 otherwise.
    /// This is an alias for `gte_scalar` to match fusor-core's API.
    /// Named `mte` (more than or equal) to match fusor-core.
    pub fn mte(&self, rhs: D) -> Self
    where
        GteOp: SimdBinaryOp<D>,
    {
        self.gte_scalar(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_eq_tensor_cpu() {
        let a: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]));
        let b: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[1.0, 3.0, 3.0, 5.0]));

        let result = a.eq_tensor(&b);
        let slice = result.as_slice().await.unwrap();

        assert_eq!(slice[[0]], 1.0); // 1 == 1
        assert_eq!(slice[[1]], 0.0); // 2 != 3
        assert_eq!(slice[[2]], 1.0); // 3 == 3
        assert_eq!(slice[[3]], 0.0); // 4 != 5
    }

    #[tokio::test]
    async fn test_lt_tensor_cpu() {
        let a: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]));
        let b: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[2.0, 2.0, 2.0, 2.0]));

        let result = a.lt_tensor(&b);
        let slice = result.as_slice().await.unwrap();

        assert_eq!(slice[[0]], 1.0); // 1 < 2
        assert_eq!(slice[[1]], 0.0); // 2 < 2
        assert_eq!(slice[[2]], 0.0); // 3 < 2
        assert_eq!(slice[[3]], 0.0); // 4 < 2
    }

    #[tokio::test]
    async fn test_gt_tensor_cpu() {
        let a: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]));
        let b: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[2.0, 2.0, 2.0, 2.0]));

        let result = a.gt_tensor(&b);
        let slice = result.as_slice().await.unwrap();

        assert_eq!(slice[[0]], 0.0); // 1 > 2
        assert_eq!(slice[[1]], 0.0); // 2 > 2
        assert_eq!(slice[[2]], 1.0); // 3 > 2
        assert_eq!(slice[[3]], 1.0); // 4 > 2
    }

    #[tokio::test]
    async fn test_lte_tensor_cpu() {
        let a: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]));
        let b: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[2.0, 2.0, 2.0, 2.0]));

        let result = a.lte_tensor(&b);
        let slice = result.as_slice().await.unwrap();

        assert_eq!(slice[[0]], 1.0); // 1 <= 2
        assert_eq!(slice[[1]], 1.0); // 2 <= 2
        assert_eq!(slice[[2]], 0.0); // 3 <= 2
        assert_eq!(slice[[3]], 0.0); // 4 <= 2
    }

    #[tokio::test]
    async fn test_gte_tensor_cpu() {
        let a: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]));
        let b: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[2.0, 2.0, 2.0, 2.0]));

        let result = a.gte_tensor(&b);
        let slice = result.as_slice().await.unwrap();

        assert_eq!(slice[[0]], 0.0); // 1 >= 2
        assert_eq!(slice[[1]], 1.0); // 2 >= 2
        assert_eq!(slice[[2]], 1.0); // 3 >= 2
        assert_eq!(slice[[3]], 1.0); // 4 >= 2
    }

    #[tokio::test]
    async fn test_eq_scalar_cpu() {
        let a: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[1.0, 2.0, 2.0, 4.0]));

        let result = a.eq_scalar(2.0);
        let slice = result.as_slice().await.unwrap();

        assert_eq!(slice[[0]], 0.0);
        assert_eq!(slice[[1]], 1.0);
        assert_eq!(slice[[2]], 1.0);
        assert_eq!(slice[[3]], 0.0);
    }

    #[tokio::test]
    async fn test_lt_scalar_cpu() {
        let a: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([4], &[1.0, 2.0, 3.0, 4.0]));

        let result = a.lt_scalar(2.5);
        let slice = result.as_slice().await.unwrap();

        assert_eq!(slice[[0]], 1.0); // 1 < 2.5
        assert_eq!(slice[[1]], 1.0); // 2 < 2.5
        assert_eq!(slice[[2]], 0.0); // 3 < 2.5
        assert_eq!(slice[[3]], 0.0); // 4 < 2.5
    }
}
