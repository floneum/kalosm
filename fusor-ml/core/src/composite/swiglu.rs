use std::sync::Arc;

use crate::{
    DataTypeEnum, Tensor,
    nary_wise::NaryExpr,
    tensor::{LazyTensorData, TensorInfo},
};

impl Tensor<3, f32> {
    pub fn swiglu_split(&self, intermediate_size: usize) -> Self {
        assert_eq!(
            self.shape()[2],
            intermediate_size * 2,
            "swiglu_split expects the last dimension to be 2 * intermediate_size"
        );
        let shape = [self.shape()[0], self.shape()[1], intermediate_size];
        let gate_index = NaryExpr::unary_op(
            NaryExpr::DimIndex(2),
            "add_intermediate",
            format!("let output = input + {intermediate_size}u;"),
            DataTypeEnum::U32,
            DataTypeEnum::U32,
        );
        let state = NaryExpr::indexed_input(
            0,
            vec![
                NaryExpr::DimIndex(0),
                NaryExpr::DimIndex(1),
                NaryExpr::DimIndex(2),
            ],
        );
        let gate = NaryExpr::indexed_input(
            0,
            vec![NaryExpr::DimIndex(0), NaryExpr::DimIndex(1), gate_index],
        );
        let gate = NaryExpr::unary_op(
            gate,
            "silu",
            "let output = input / (1.0 + exp(-input));",
            DataTypeEnum::F32,
            DataTypeEnum::F32,
        );
        let expression = NaryExpr::mul(state, gate, DataTypeEnum::F32);
        let operation = crate::nary_wise::NaryOperation {
            inputs: vec![self.key()],
            expression,
            shape: shape.into(),
            output_datatype: DataTypeEnum::F32,
        };

        let device = self.device().clone();
        let info = TensorInfo::new(shape.into(), DataTypeEnum::F32);
        let key = device.compute_graph().create_custom(Arc::new(operation));
        Tensor::from_parts(LazyTensorData::from_parts(device, info, key))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_swiglu_split_matches_composite() {
    use crate::Device;

    let device = Device::test_instance();
    let data = [[
        [1.0, -2.0, 0.5, -0.25, 0.1, 1.5],
        [3.0, 0.25, -1.0, 2.0, -0.5, 0.75],
    ]];
    let tensor = Tensor::new(&device, &data);
    let fused = tensor.swiglu_split(3);
    let state = tensor.narrow(2, 0, 3);
    let gate = tensor.narrow(2, 3, 3).silu();
    let expected = state * gate;

    let fused = fused.as_slice().await.unwrap();
    let expected = expected.as_slice().await.unwrap();
    for (actual, expected) in fused.as_slice().iter().zip(expected.as_slice()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}
