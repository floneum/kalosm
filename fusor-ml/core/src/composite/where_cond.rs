use crate::{
    Tensor,
    nary_wise::{NaryExpr, NaryOperation},
};

impl Tensor {
    pub fn where_cond(self, on_true: &Tensor, on_false: &Tensor) -> Tensor {
        assert_eq!(on_true.shape(), on_false.shape());
        assert_eq!(on_true.datatype(), on_false.datatype());
        assert_eq!(self.shape(), on_true.shape());
        let shape: Box<[usize]> = self.shape().into();
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
        let info = crate::tensor::TensorInfo::new(on_true.shape().into(), on_true.datatype());
        let key = device.compute_graph().create_nary(nary);
        Tensor::from_parts(crate::tensor::LazyTensorData::from_parts(device, info, key))
    }
}
