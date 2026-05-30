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

impl Tensor {
    pub fn index_select(&self, dimension: usize, indexes: &Tensor) -> Self {
        indexes.assert_rank::<1>();
        indexes.assert_datatype::<u32>();
        assert!(dimension < self.rank());
        let output_shape = index_select_output_shape(dimension, self.shape(), indexes.shape());
        let nary = NaryOperation {
            inputs: vec![self.key(), indexes.key()],
            expression: NaryExpr::index_select(self.rank(), dimension),
            shape: output_shape.clone(),
            output_datatype: self.datatype(),
        };
        let device = self.device().clone();
        let info = crate::tensor::TensorInfo::new(output_shape, self.datatype());
        let key = device.compute_graph().create_nary(nary);
        Self::from_parts(crate::tensor::LazyTensorData::from_parts(device, info, key))
    }
}
