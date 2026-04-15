use crate::{
    Tensor,
    nary_wise::{NaryExpr, NaryOperation},
};

/// Compute the output shape for an index_select operation.
pub(crate) fn index_select_output_shape(
    dimension: usize,
    value_shape: &[usize],
    indexes_shape: &[usize],
) -> Box<[usize]> {
    value_shape
        .iter()
        .enumerate()
        .map(|(i, dim)| {
            if i == dimension {
                indexes_shape[0]
            } else {
                *dim
            }
        })
        .collect()
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn index_select(&self, dimension: usize, indexes: &Tensor<1, u32>) -> Self {
        assert!(dimension < R);
        let output_shape = index_select_output_shape(dimension, self.shape(), indexes.shape());
        let nary = NaryOperation {
            inputs: vec![self.key(), indexes.key()],
            expression: NaryExpr::index_select(R, dimension),
            shape: output_shape.clone(),
            output_datatype: T::WGSL_TYPE,
        };
        let device = self.device().clone();
        let info = crate::tensor::TensorInfo::new(output_shape, T::WGSL_TYPE);
        let key = device.compute_graph().create_nary(nary);
        Self::from_parts(crate::tensor::LazyTensorData::from_parts(device, info, key))
    }
}

