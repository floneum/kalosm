use std::ops::Range;

use crate::composite::index::IndexOp;

use super::*;

impl<const R: usize> Tensor<R> {
    pub fn index_select(&self, dimension: usize, indices: &RawTensor<1, u32>) -> Self {
        let input_shape = self.shape();
        assert!(dimension < R, "index_select dimension out of bounds");

        let value = self.value.index_select(dimension, indices).to_concrete();
        let input_id = self.handle.id;
        let indices = indices.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "index_select")?;
            let one_hot = one_hot_matrix(&indices, input_shape[dimension]);
            // transpose+reshape only commute through a copy, so the moved axis
            // is materialized once on each side of the matmul; dimension 0
            // needs neither.
            let moved = if dimension == 0 {
                gradient
            } else {
                gradient.transpose(0, dimension).to_concrete()
            };
            let moved_shape = moved.shape();
            let rest = moved_shape[1..].iter().product::<usize>();
            let flat = moved.reshape([moved_shape[0], rest]);
            let scattered = one_hot.transpose(0, 1).mat_mul(&flat);
            let mut unmoved_shape = moved_shape;
            unmoved_shape[0] = input_shape[dimension];
            let scattered = scattered.reshape(unmoved_shape);
            let input_gradient = if dimension == 0 {
                scattered.to_concrete()
            } else {
                scattered.to_concrete().transpose(0, dimension).to_concrete()
            };
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(input_gradient),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn index_ops<const OUT: usize>(&self, ops: [IndexOp; R]) -> Tensor<OUT>
    where
        crate::gpu::Tensor<R, f32>: crate::gpu::SmallerRank<1, OUT, f32>,
    {
        let shape = self.shape();
        let slices: [Range<usize>; R] =
            std::array::from_fn(|axis| ops[axis].to_range(shape[axis]));
        let dim = crate::composite::index::removed_dim(ops.map(|op| op.removes_dim()));
        self.slice(slices).squeeze_dims::<1, OUT>([dim])
    }
}

impl Tensor<2> {
    pub fn i<I1, I2>(&self, index: (I1, I2)) -> Tensor<1>
    where
        I1: Into<IndexOp>,
        I2: Into<IndexOp>,
    {
        self.index_ops([index.0.into(), index.1.into()])
    }

    pub fn gather_last(&self, indices: &RawTensor<1, u32>) -> Tensor<1> {
        let shape = self.shape();
        assert_eq!(
            shape[0],
            indices.shape()[0],
            "gather_last expects one index per row"
        );
        let width = shape[1];
        let device = self.device();
        let row_offsets = (0..shape[0])
            .map(|row| (row * width) as u32)
            .collect::<Vec<_>>();
        let row_offsets: RawTensor<1, u32> =
            RawTensor::from_slice(&device, [shape[0]], &row_offsets);
        let linear_indices = (row_offsets + indices.clone()).to_concrete();
        let flat = self.value.reshape([shape[0] * width]).to_concrete();
        let value = flat.index_select(0, &linear_indices).to_concrete();
        let input_id = self.handle.id;
        let indices = indices.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<1>(&*gradient, "gather_last")?;
            let one_hot = one_hot_matrix(&indices, width);
            let input_gradient: RawTensor<2, f32> = one_hot.mul_(&gradient.reshape([shape[0], 1]));
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(input_gradient),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    /// Numerically stable softmax cross-entropy against integer class
    /// targets, averaged over rows: `mean_i(LSE(x_i) - x_i[t_i])` with the
    /// log-sum-exp max-shifted. The backward is analytic —
    /// `dlogits = (softmax(x) - onehot(t)) * grad / rows` — so the whole
    /// loss runs in a handful of fused kernels instead of a taped chain.
    pub fn softmax_cross_entropy(&self, targets: &RawTensor<1, u32>) -> Tensor<0> {
        let [rows, width] = self.shape();
        assert_eq!(
            targets.shape()[0],
            rows,
            "softmax_cross_entropy expects one target per row"
        );
        let device = self.value.device();
        let logits = self.value.clone();

        // Forward: the shifted exp-sum chain is exclusively consumed, so it
        // fuses into one row program; the label logits gather reads the raw
        // logits directly.
        let max = logits.max_keepdim::<1>(1);
        let shifted_exp = (&logits - &max.broadcast_as([rows, width]))
            .to_concrete()
            .exp()
            .to_concrete();
        let lse_total = (shifted_exp.sum_keepdim::<1>(1).to_concrete().log().to_concrete() + max)
            .to_concrete();
        let row_offsets: Vec<u32> = (0..rows).map(|row| (row * width) as u32).collect();
        let row_offsets: RawTensor<1, u32> = RawTensor::from_slice(&device, [rows], &row_offsets);
        let linear = (row_offsets + targets.clone()).to_concrete();
        let flat = logits.reshape([rows * width]).to_concrete();
        let picked = flat.index_select(0, &linear).to_concrete();
        let per_row = (lse_total.reshape([rows]).to_concrete() - picked).to_concrete();
        let value = (per_row.sum::<0>(0) * (1.0 / rows as f32)).to_concrete();

        let input_id = self.handle.id;
        let logits_value = self.value.clone();
        let targets = targets.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let grad = downcast_tensor::<0>(&*gradient, "softmax_cross_entropy")?;
            let probs = logits_value.softmax_last_dim::<1>();
            let one_hot = one_hot_matrix(&targets, width);
            let scale = (grad.reshape([1, 1]).to_concrete() * (1.0 / rows as f32)).to_concrete();
            let diff = (probs - one_hot).to_concrete();
            let dlogits = (&diff * &scale.broadcast_as([rows, width])).to_concrete();
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(dlogits),
            }])
        });
        self.emit_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn embedding(&self, indices: &RawTensor<2, u32>) -> Tensor<3> {
        let [rows, columns] = indices.shape();
        let width = self.shape()[1];
        let flat_indices = indices.clone().reshape([rows * columns]).to_concrete();
        self.index_select(0, &flat_indices)
            .reshape([rows, columns, width])
    }
}

impl Tensor<3> {
    pub fn i<I1, I2, I3>(&self, index: (I1, I2, I3)) -> Tensor<2>
    where
        I1: Into<IndexOp>,
        I2: Into<IndexOp>,
        I3: Into<IndexOp>,
    {
        self.index_ops([index.0.into(), index.1.into(), index.2.into()])
    }
}

impl Tensor<4> {
    pub fn i<I1, I2, I3, I4>(&self, index: (I1, I2, I3, I4)) -> Tensor<3>
    where
        I1: Into<IndexOp>,
        I2: Into<IndexOp>,
        I3: Into<IndexOp>,
        I4: Into<IndexOp>,
    {
        self.index_ops([
            index.0.into(),
            index.1.into(),
            index.2.into(),
            index.3.into(),
        ])
    }
}

/// Build a `[indices.len(), size]` f32 matrix with 1.0 at `[row, indices[row]]`
/// so scatter-adds stay on-device as matmuls/products against it; duplicate
/// indices accumulate through the contraction.
fn one_hot_matrix(indices: &RawTensor<1, u32>, size: usize) -> RawTensor<2, f32> {
    let device = indices.device();
    let rows = indices.shape()[0];
    let positions = (0..size)
        .map(|position| position as f32)
        .collect::<Vec<_>>();
    let positions: RawTensor<2, f32> = RawTensor::from_slice(&device, [1, size], &positions);
    let indices = indices.cast::<f32>().reshape([rows, 1]).to_concrete();
    indices.sub_(&positions).eq(0.0)
}
