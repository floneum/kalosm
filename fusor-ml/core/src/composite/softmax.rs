use crate::Tensor;

impl Tensor {
    /// Softmax along `axis`, expressed as its composed form:
    /// `exp(x - max(x)) / sum(exp(x - max(x)))` over keepdim-broadcast
    /// reductions. The resolver recognizes the canonical cluster and routes
    /// it to the fused softmax kernels; ineligible shapes lower through the
    /// generic elementwise + reduce path (the same math).
    pub fn softmax(&self, axis: usize) -> Self {
        assert!(axis < self.rank(), "softmax axis out of bounds");
        let mut kept_shape = self.shape().to_vec();
        kept_shape[axis] = 1;
        let max = self
            .max(axis)
            .reshape(&kept_shape)
            .broadcast_as(self.shape());
        let shifted = self - &max;
        let exp = shifted.exp();
        let sum = exp
            .sum(axis)
            .reshape(&kept_shape)
            .broadcast_as(self.shape());
        &exp / &sum
    }

    pub fn softmax_last_dim(&self) -> Self {
        let Some(axis) = self.rank().checked_sub(1) else {
            panic!("softmax_last_dim requires rank >= 1");
        };
        self.softmax(axis)
    }
}
