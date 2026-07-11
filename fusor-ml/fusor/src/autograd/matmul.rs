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
        self.emit_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    pub fn mat_mul(&self, rhs: &Self) -> Self {
        self.mat_mul_internal(rhs)
    }

    /// `self @ rhs^T` (transposing the last two dims of `rhs`), with a
    /// backward that produces the `rhs` gradient directly in `rhs`'s own
    /// layout: `d_rhs = grad^T @ self` is contiguous, so optimizer-side
    /// flattens stay zero-cost views instead of gather kernels.
    pub fn mat_mul_transposed_rhs(&self, rhs: &Self) -> Self {
        assert_same_graph(self, rhs);
        let value = self.value.mat_mul(&rhs.value.transpose(R - 2, R - 1));
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "mat_mul_transposed_rhs")?;
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(gradient.clone().mat_mul(&rhs_value)),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(gradient.transpose(R - 2, R - 1).mat_mul(&lhs_value)),
                },
            ])
        });
        self.emit_op(
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
        if R == 1 {
            let k = self.shape()[0];
            let n = weights.shape()[0];
            let out_shape: [usize; R] = std::array::from_fn(|_| n);
            return self.reshape([1, k]).q_mat_mul(weights).reshape(out_shape);
        }
        let value = self.value.q_mat_mul(weights).into_concrete();
        if !self.requires_grad() {
            return self.emit_op(value, vec![self.handle.clone()], None);
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
                .into_concrete();
            let weight = Tensor::constant_from_raw(&input.graph(), weight);
            input.mat_mul_internal(&weight)
        })
    }
}
