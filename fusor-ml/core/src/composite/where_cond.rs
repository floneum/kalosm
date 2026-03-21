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
            shape: shape,
            output_datatype: on_true.datatype(),
        };
        let device = on_true.device().clone();
        let info = crate::tensor::TensorInfo::new(
            on_true.shape().as_slice().into(),
            on_true.datatype(),
        );
        let key = device.compute_graph().create_nary(nary);
        Tensor::from_parts(crate::tensor::LazyTensorData::from_parts(device, info, key))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_where_cond() {
    use crate::Device;

    let device = Device::test_instance();

    let data_vec_f32: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let data = Tensor::new(&device, &data_vec_f32);
    let data_vec_u32: Vec<u32> = (0..10).collect();
    let even = Tensor::new(&device, &data_vec_u32) % 2;
    let zero = Tensor::splat(&device, 0., *data.shape());

    let data_where_even = even.where_cond(&data, &zero);

    let result = data_where_even.as_slice().await.unwrap();
    println!("result: {result:?}");

    assert_eq!(result[[0]], 0.);
    assert_eq!(result[[1]], 1.);
    assert_eq!(result[[2]], 0.);
    assert_eq!(result[[3]], 3.);
    assert_eq!(result[[4]], 0.);
    assert_eq!(result[[5]], 5.);
    assert_eq!(result[[6]], 0.);
    assert_eq!(result[[7]], 7.);
    assert_eq!(result[[8]], 0.);
    assert_eq!(result[[9]], 9.);
}
