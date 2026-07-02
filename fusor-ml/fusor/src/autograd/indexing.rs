use crate::layers::Embedding;

use super::*;

impl Tensor<2> {
    pub fn index_select(&self, dimension: usize, indices: &RawTensor<1, u32>) -> Tensor<2> {
        let input_shape = self.shape();
        assert!(dimension < 2, "index_select dimension out of bounds");

        let value = self.value.index_select(dimension, indices).to_concrete();
        let input_id = self.handle.id;
        let indices = indices.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "index_select")?;
            let one_hot = one_hot_matrix(&indices, input_shape[dimension]);
            let input_gradient = if dimension == 0 {
                one_hot.transpose(0, 1).mat_mul(&gradient)
            } else {
                gradient.mat_mul(&one_hot)
            };
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(input_gradient),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
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
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn embedding(&self, indices: &RawTensor<2, u32>) -> Tensor<3> {
        let value: RawTensor<3, f32> =
            Embedding::new_from_tensor(self.value.clone()).forward(indices);
        let table_id = self.handle.id;
        let table_shape = self.shape();
        let indices = indices.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "embedding")?;
            let grad_shape = gradient.shape();
            let rows = grad_shape[0] * grad_shape[1];
            let grad_flat = gradient.reshape([rows, grad_shape[2]]).to_concrete();
            let flat_indices = indices.reshape([rows]).to_concrete();
            let one_hot = one_hot_matrix(&flat_indices, table_shape[0]);
            Ok(vec![BackwardTarget {
                node: table_id,
                gradient: Box::new(one_hot.transpose(0, 1).mat_mul(&grad_flat)),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
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
