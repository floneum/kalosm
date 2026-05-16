use std::ops::Range;

use crate::{
    DataTypeEnum, TILE_SIZE, Tensor, TensorData,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{
        kernel_backend::DirectKernel,
        inputs::MirValue,
        operation::Operation,
        workgroup_shape::{WorkgroupShape, WorkgroupShapeConstraints},
    },
    nary_wise::{NaryExpr, NaryOp, NaryOperation, NaryScalar},
    visit_tiled::{titled_map_dispatch_size, titled_map_workgroup_size_constraints},
};

#[derive(Clone, Debug)]
pub(crate) struct SliceAssignOperation {
    pub(crate) input: NodeIndex,
    pub(crate) value: NodeIndex,
    pub(crate) slices: Box<[Range<usize>]>,
    pub(crate) input_shape: Box<[usize]>,
    pub(crate) in_place: bool,
}

impl SliceAssignOperation {
    pub fn new(
        input: NodeIndex,
        value: NodeIndex,
        slices: Box<[Range<usize>]>,
        input_shape: Box<[usize]>,
    ) -> Self {
        Self {
            input,
            value,
            slices,
            input_shape,
            in_place: false,
        }
    }

    pub fn new_in_place(
        input: NodeIndex,
        value: NodeIndex,
        slices: Box<[Range<usize>]>,
        input_shape: Box<[usize]>,
    ) -> Self {
        Self {
            input,
            value,
            slices,
            input_shape,
            in_place: true,
        }
    }

    fn value_shape(&self) -> Box<[usize]> {
        self.slices
            .iter()
            .map(|slice| slice.end - slice.start)
            .collect()
    }

    fn operation_shape(&self) -> Box<[usize]> {
        if self.in_place {
            self.value_shape()
        } else {
            self.input_shape.clone()
        }
    }

    fn expression(&self, datatype: DataTypeEnum) -> NaryExpr {
        if self.in_place {
            return NaryExpr::input(0, self.slices.len());
        }

        let rank = self.slices.len();
        let mut condition = NaryExpr::scalar(NaryScalar::U32(1));
        for (dim, slice) in self.slices.iter().enumerate() {
            let dim_index = NaryExpr::DimIndex(dim);
            let ge_start = NaryExpr::unary_op(
                dim_index.clone(),
                "ge_start",
                NaryOp::GreaterEqualConst(NaryScalar::U32(slice.start as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            let lt_end = NaryExpr::unary_op(
                dim_index,
                "lt_end",
                NaryOp::LessConst(NaryScalar::U32(slice.end as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            condition = NaryExpr::mul(condition, ge_start, DataTypeEnum::U32);
            condition = NaryExpr::mul(condition, lt_end, DataTypeEnum::U32);
        }

        let value_indices = self
            .slices
            .iter()
            .enumerate()
            .map(|(dim, slice)| {
                let shifted_index = if slice.start == 0 {
                    NaryExpr::DimIndex(dim)
                } else {
                    NaryExpr::unary_op(
                        NaryExpr::DimIndex(dim),
                        "slice_offset",
                        NaryOp::SubConst(NaryScalar::U32(slice.start as u32)),
                        DataTypeEnum::U32,
                        DataTypeEnum::U32,
                    )
                };
                NaryExpr::select(
                    condition.clone(),
                    shifted_index,
                    NaryExpr::scalar(NaryScalar::U32(0)),
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                )
            })
            .collect();

        NaryExpr::select(
            condition,
            NaryExpr::indexed_input(1, value_indices),
            NaryExpr::input(0, rank),
            DataTypeEnum::U32,
            datatype,
        )
    }
}

impl Operation for SliceAssignOperation {
    fn workgroup_shape_constraints(&self, device: &crate::Device) -> WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(&self.operation_shape(), device)
    }

    fn dispatch_size(&self, workgroup_shape: &WorkgroupShape, _inputs: &[MirValue]) -> [u32; 3] {
        titled_map_dispatch_size(TILE_SIZE, *workgroup_shape, &self.operation_shape())
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.value);
        f(self.input);
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        // Pass the ORIGINAL input tensor (not sliced) and the value tensor
        let input = nodes.get_cached_result(self.input).unwrap();
        let value = nodes.get_cached_result(self.value).unwrap();

        if self.in_place {
            let output = input.slice(&self.slices);
            return vec![input.clone().into(), value.clone().into(), output.into()];
        }

        // Create output buffer with the same shape as input
        let output =
            TensorData::new_for_shape(input.device(), input.layout().shape(), input.datatype());

        vec![input.clone().into(), value.clone().into(), output.into()]
    }

    fn build_direct_kernel(
        &self,
        graph: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        if self.in_place {
            let value = inputs[1].as_tensor()?;
            let operation = NaryOperation {
                inputs: vec![self.value],
                expression: self.expression(value.datatype()),
                shape: value.layout().shape().into(),
                output_datatype: value.datatype(),
            };
            return crate::nary_direct::build_nary_direct_kernel_to_output(
                &operation,
                graph,
                workgroup_shape,
                &[inputs[1].clone(), inputs[2].clone()],
                1,
            );
        }

        let input = inputs[0].as_tensor()?;
        let operation = NaryOperation {
            inputs: vec![self.input, self.value],
            expression: self.expression(input.datatype()),
            shape: self.input_shape.clone(),
            output_datatype: input.datatype(),
        };
        crate::nary_direct::build_nary_direct_kernel_to_output(
            &operation,
            graph,
            workgroup_shape,
            inputs,
            2,
        )
    }

    fn requires_single_kernel_batch(&self) -> bool {
        true
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        if self.in_place {
            return inputs[0].clone();
        }

        // Return the output tensor (last input)
        inputs[2].clone()
    }

    fn name(&self) -> String {
        format!(
            "slice_assign_{}",
            self.slices
                .iter()
                .map(|slice| format!("{slice:?}"))
                .collect::<Vec<_>>()
                .join("_")
        )
    }
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn slice_assign(&self, slices: [Range<usize>; R], value: &Self) -> Self {
        self.add_slice_assign(value, slices)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};

    #[tokio::test]
    async fn slice_assign_in_place_updates_only_slice() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let base_rows = vec![vec![0.0f32; 4]; 3];
        let value_rows = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let base: Tensor<2, f32> = Tensor::new(&device, &base_rows);
        let value: Tensor<2, f32> = Tensor::new(&device, &value_rows);

        let updated = base.slice_assign_in_place([1..3, 1..3], &value);
        let updated = updated.as_slice().await.unwrap();

        assert_eq!(updated.shape(), &[3, 4]);
        assert_eq!(updated[[0, 0]], 0.0);
        assert_eq!(updated[[1, 0]], 0.0);
        assert_eq!(updated[[1, 1]], 1.0);
        assert_eq!(updated[[1, 2]], 2.0);
        assert_eq!(updated[[2, 1]], 3.0);
        assert_eq!(updated[[2, 2]], 4.0);
        assert_eq!(updated[[2, 3]], 0.0);
    }
}
