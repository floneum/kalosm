use std::{hash::Hash, ops::Range};

use crate::{
    DataTypeEnum, TILE_SIZE, Tensor,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{
        inputs::MirValue,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{WorkgroupShape, WorkgroupShapeConstraints},
    },
    nary_wise::{ElementwiseOperation, NaryExpr, NaryOp, NaryScalar},
    visit_tiled::{titled_map_dispatch_size, titled_map_workgroup_size_constraints},
};

/// The in-place region write of the graph vocabulary: dispatch over the
/// slice region only, writing the value into the input's buffer. The
/// out-of-place form has no operation — `Tensor::slice_assign` composes it
/// as a plain elementwise select (see [`slice_assign_expression`]).
#[derive(Clone, Debug)]
pub(crate) struct SliceAssignOperation {
    pub(crate) input: NodeIndex,
    pub(crate) value: NodeIndex,
    pub(crate) slices: Box<[Range<usize>]>,
}

/// The region predicate of a slice assign: 1 when every output coordinate
/// lies inside its slice range, 0 otherwise.
pub(crate) fn slice_region_condition(slices: &[Range<usize>]) -> NaryExpr {
    let mut condition = NaryExpr::scalar(NaryScalar::U32(1));
    for (dim, slice) in slices.iter().enumerate() {
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
    condition
}

/// The composed slice-assign body: per output coordinate, read the assigned
/// value inside the slice region and the original input outside it. Inputs:
/// 0 = the original tensor, 1 = the assigned value.
pub(crate) fn slice_assign_expression(slices: &[Range<usize>], datatype: DataTypeEnum) -> NaryExpr {
    let rank = slices.len();
    let condition = slice_region_condition(slices);

    let value_indices = slices
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

impl SliceAssignOperation {
    pub fn new_in_place(input: NodeIndex, value: NodeIndex, slices: Box<[Range<usize>]>) -> Self {
        Self {
            input,
            value,
            slices,
        }
    }

    fn value_shape(&self) -> Box<[usize]> {
        self.slices
            .iter()
            .map(|slice| slice.end - slice.start)
            .collect()
    }
}

impl Operation for SliceAssignOperation {
    fn hash_kernel_fields(&self, state: &mut rustc_hash::FxHasher) {
        self.slices.hash(state);
    }

    fn workgroup_shape_constraints(&self, device: &crate::Device) -> WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(&self.value_shape(), device)
    }

    fn dispatch_size(&self, workgroup_shape: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3] {
        let max_per_dim = inputs[0]
            .as_tensor()
            .unwrap()
            .device()
            .limits()
            .max_compute_workgroups_per_dimension;
        titled_map_dispatch_size(
            TILE_SIZE,
            *workgroup_shape,
            &self.value_shape(),
            max_per_dim,
        )
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.value);
        f(self.input);
    }

    fn visit_dependencies_mut(&mut self, f: &mut dyn FnMut(&mut NodeIndex)) {
        f(&mut self.value);
        f(&mut self.input);
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        // Pass the ORIGINAL input tensor (not sliced) and the value tensor
        let input = nodes.get_cached_result(self.input).unwrap();
        let value = nodes.get_cached_result(self.value).unwrap();
        let output = input.slice(&self.slices);
        vec![input.clone().into(), value.clone().into(), output.into()]
    }

    fn build_direct_kernel(
        &self,
        graph: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        // A copy kernel over the slice region: read the value, write into
        // the sliced view of the input's buffer.
        let value = inputs[1].as_tensor()?;
        let operation = ElementwiseOperation {
            inputs: vec![self.value],
            expression: NaryExpr::input(0, self.slices.len()),
            shape: value.layout().shape().into(),
            output_datatype: value.datatype(),
        };
        crate::nary_direct::build_nary_direct_kernel_to_output(
            &operation,
            graph,
            workgroup_shape,
            &[inputs[1].clone(), inputs[2].clone()],
            1,
        )
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs[0].clone()
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

impl Tensor {
    pub fn slice_assign(&self, slices: impl Into<Box<[Range<usize>]>>, value: &Self) -> Self {
        self.add_slice_assign(value, slices)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};

    #[test]
    fn slice_assign_in_place_updates_only_slice() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let base_rows = vec![vec![0.0f32; 4]; 3];
            let value_rows = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
            let base = Tensor::new::<f32, 2, _>(&device, &base_rows);
            let value = Tensor::new::<f32, 2, _>(&device, &value_rows);

            let updated = base.slice_assign_in_place([1..3, 1..3], &value);
            let updated = updated.as_slice::<2, f32>().await.unwrap();

            assert_eq!(updated.shape(), &[3, 4]);
            assert_eq!(updated[[0, 0]], 0.0);
            assert_eq!(updated[[1, 0]], 0.0);
            assert_eq!(updated[[1, 1]], 1.0);
            assert_eq!(updated[[1, 2]], 2.0);
            assert_eq!(updated[[2, 1]], 3.0);
            assert_eq!(updated[[2, 2]], 4.0);
            assert_eq!(updated[[2, 3]], 0.0);
        });
    }
}
