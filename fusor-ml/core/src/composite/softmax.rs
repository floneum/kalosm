use std::sync::Arc;

use crate::{
    Tensor,
    softmax::SoftmaxOperation,
    tensor::{LazyTensorData, TensorInfo},
};

impl Tensor {
    pub fn softmax(&self, axis: usize) -> Self {
        assert!(axis < self.rank(), "softmax axis out of bounds");
        if let Some(operation) = SoftmaxOperation::new(
            self.key(),
            self.shape(),
            axis,
            self.datatype(),
            self.device(),
        ) {
            let device = self.device().clone();
            let info = TensorInfo::new(self.shape().into(), self.datatype());
            let key = device.compute_graph().create_graph_op(Arc::new(operation));
            return Tensor::from_parts(LazyTensorData::from_parts(device, info, key));
        }

        self.softmax_composite(axis)
    }

    fn softmax_composite(&self, axis: usize) -> Self {
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
