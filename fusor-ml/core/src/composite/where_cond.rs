use crate::{
    DataType, Tensor,
    nary_wise::{NaryExpr, NaryOperation},
};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn where_cond<D2>(self, on_true: &Tensor<R, D2>, on_false: &Tensor<R, D2>) -> Tensor<R, D2>
    where
        D2: DataType,
    {
        let shape: Box<[usize]> = self.shape().as_slice().into();
        let rank = shape.len();
        let nary = NaryOperation {
            inputs: vec![self.key(), on_true.key(), on_false.key()],
            expression: NaryExpr::select(
                NaryExpr::input(0, rank),
                NaryExpr::input(1, rank),
                NaryExpr::input(2, rank),
                self.datatype(),
                on_true.datatype(),
            ),
            shape,
            output_datatype: on_true.datatype(),
        };
        let device = on_true.device().clone();
        let info =
            crate::tensor::TensorInfo::new(on_true.shape().as_slice().into(), on_true.datatype());
        let key = device.compute_graph().create_nary(nary);
        Tensor::from_parts(crate::tensor::LazyTensorData::from_parts(device, info, key))
    }
}
