use super::*;

impl<const R: usize> Tensor<R> {
    pub(super) fn mat_mul_internal(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        let value = self.value.mat_mul(&rhs.value);
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "mat_mul")?;
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(
                        gradient.clone().mat_mul(&rhs_value.transpose(R - 2, R - 1)),
                    ),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(lhs_value.transpose(R - 2, R - 1).mat_mul(&gradient)),
                },
            ])
        });
        self.from_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        self.mat_mul_internal(rhs)
    }

    pub fn t(&self) -> Self {
        assert!(R >= 2, "t requires rank >= 2");
        self.transpose(R - 2, R - 1)
    }

    pub fn q_mat_mul(&self, weights: &crate::QMatrix) -> Self {
        assert!(R >= 2, "q_mat_mul requires rank >= 2");
        let value = self.value.q_mat_mul(weights).to_concrete();
        if !self.requires_grad() {
            return self.from_op(value, vec![self.handle.clone()], None);
        }
        let weights = weights.clone();
        self.replay_unary("q_mat_mul", value, move |input| {
            let n = weights.shape()[0];
            let k = weights.shape()[1];
            let batch_dims = R - 2;
            let weight_shape: [usize; R] = std::array::from_fn(|i| {
                if i < batch_dims {
                    1
                } else if i == batch_dims {
                    k
                } else {
                    n
                }
            });
            let dequantized: RawTensor<2, f32> = weights.dequantize();
            let weight = dequantized
                .transpose(0, 1)
                .reshape(weight_shape)
                .to_concrete();
            let weight = Tensor::constant_from_raw(&input.graph(), weight);
            input.mat_mul_internal(&weight)
        })
    }
}

impl Tensor<2> {
    pub fn mat_mul(&self, rhs: &Tensor<2>) -> Tensor<2> {
        self.mat_mul_internal(rhs)
    }
}

impl Tensor<3> {
    pub fn mat_mul(&self, rhs: &Tensor<3>) -> Tensor<3> {
        self.mat_mul_internal(rhs)
    }
}
