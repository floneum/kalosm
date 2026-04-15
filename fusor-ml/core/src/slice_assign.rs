use std::{fmt::Write, ops::Range};

use crate::{
    TILE_SIZE, Tensor, TensorData,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{
        inputs::MirValue,
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{WorkgroupShape, WorkgroupShapeConstraints},
    },
    visit_tiled::{
        MaybeQTensorInput, VisitTiledInput, build_visit_tiled_kernel, titled_map_dispatch_size,
        titled_map_workgroup_size_constraints,
    },
};

#[derive(Clone, Debug)]
pub(crate) struct SliceAssignOperation {
    pub(crate) input: NodeIndex,
    pub(crate) value: NodeIndex,
    pub(crate) slices: Box<[Range<usize>]>,
    pub(crate) input_shape: Box<[usize]>,
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
        }
    }
}

impl Operation for SliceAssignOperation {
    fn workgroup_shape_constraints(&self, device: &crate::Device) -> WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(&self.input_shape, device)
    }

    fn dispatch_size(&self, workgroup_shape: &WorkgroupShape, _inputs: &[MirValue]) -> [u32; 3] {
        titled_map_dispatch_size(TILE_SIZE, *workgroup_shape, &self.input_shape)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.value);
        f(self.input);
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        // Pass the ORIGINAL input tensor (not sliced) and the value tensor
        let input = nodes.get_cached_result(self.input).unwrap();
        let value = nodes.get_cached_result(self.value).unwrap();

        // Create output buffer with the same shape as input
        let output =
            TensorData::new_for_shape(input.device(), input.layout().shape(), input.datatype());

        vec![input.clone().into(), value.clone().into(), output.into()]
    }

    fn build_kernel(
        &self,
        graph: &ComputeGraphInner,
        _workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        let input = inputs[0].as_tensor().unwrap();
        let value = inputs[1].as_tensor().unwrap();
        let _output = inputs[2].as_tensor().unwrap();
        let dtype = input.datatype();
        let rank = self.slices.len() as u32;
        let value_rank = value.layout().shape().len() as u32;

        // Build inputs for visit_tiled: input, value, output (all same dtype)
        let tiled_inputs = vec![
            VisitTiledInput::new(dtype.into(), rank),
            VisitTiledInput::new(dtype.into(), value_rank),
            VisitTiledInput::new(dtype.into(), rank),
        ];

        // Output tensor is at index 2
        let output_tensor_idx = 2;

        // Capture slice bounds for use in closure
        let slices = self.slices.clone();

        build_visit_tiled_kernel(
            &graph.device(),
            &self.input_shape,
            TILE_SIZE,
            tiled_inputs,
            output_tensor_idx,
            |kernel, indexes, tensors, values| {
                let input_value = &values[0];
                let output_index = &indexes[output_tensor_idx];
                let output_tensor = &tensors[output_tensor_idx];
                let value_tensor = &tensors[1];

                // Build the condition: slice_start <= idx < slice_end for each dimension
                let mut conditions = Vec::new();
                for dim in 0..rank as usize {
                    let start = slices[dim].start;
                    let end = slices[dim].end;
                    conditions.push(format!("(dim_{dim} >= {start}u && dim_{dim} < {end}u)"));
                }
                let in_slice_condition = conditions.join(" && ");

                writeln!(kernel, "var slice_val: {};", dtype.as_str()).unwrap();
                writeln!(kernel, "if ({}) {{", in_slice_condition).unwrap();

                // Inside slice: compute value tensor index and read from value
                for dim in 0..rank as usize {
                    let offset = slices[dim].start;
                    writeln!(
                        kernel,
                        "    let value_idx_{} = dim_{} - {}u;",
                        dim, dim, offset
                    )
                    .unwrap();
                }

                // Read from value tensor with computed indices
                write!(kernel, "    slice_val = ").unwrap();
                match value_tensor {
                    MaybeQTensorInput::Tensor(t) => {
                        write!(kernel, "{t}[").unwrap();
                        t.strided_index(
                            kernel,
                            (0..rank as usize).map(|d| format!("value_idx_{}", d)),
                        );
                        writeln!(kernel, "];").unwrap();
                    }
                    MaybeQTensorInput::QTensor(_) => {
                        panic!("Value tensor cannot be quantized")
                    }
                }

                writeln!(kernel, "}} else {{").unwrap();

                // Outside slice: copy from input
                writeln!(kernel, "    slice_val = {};", input_value).unwrap();

                writeln!(kernel, "}}").unwrap();

                // Return the assignment expression
                format!("{output_tensor}[{output_index}] = slice_val;")
            },
            kernel,
        );
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
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

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        // Output has the same layout as input
        map.get(&self.input).unwrap().clone()
    }
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn slice_assign(&self, slices: [Range<usize>; R], value: &Self) -> Self {
        self.add_slice_assign(value, slices)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_slice_assign() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let value_tensor = Tensor::new(&device, &[[10., 20.], [30., 40.]]);
    let tensor = tensor.slice_assign([0..2, 0..2], &value_tensor);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 10.);
    assert_eq!(as_slice[[0, 1]], 20.);
    assert_eq!(as_slice[[1, 0]], 30.);
    assert_eq!(as_slice[[1, 1]], 40.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_slice_assign_nonzero_offset() {
    use crate::Device;

    let device = Device::test_instance();

    // Create a 3x5 zeros tensor and assign a 2x2 value at offset [1, 3]
    let zeros: Tensor<2, f32> = Tensor::splat(&device, 0.0f32, [3, 5]);
    let value_tensor = Tensor::new(&device, &[[10., 20.], [30., 40.]]);
    let result = zeros.slice_assign([1..3, 3..5], &value_tensor);
    let as_slice = result.as_slice().await.unwrap();
    println!("{as_slice:?}");

    // First row should be all zeros
    assert_eq!(as_slice[[0, 0]], 0.);
    assert_eq!(as_slice[[0, 1]], 0.);
    assert_eq!(as_slice[[0, 2]], 0.);
    assert_eq!(as_slice[[0, 3]], 0.);
    assert_eq!(as_slice[[0, 4]], 0.);

    // Second row: zeros in cols 0-2, then 10, 20 at cols 3-4
    assert_eq!(as_slice[[1, 0]], 0.);
    assert_eq!(as_slice[[1, 1]], 0.);
    assert_eq!(as_slice[[1, 2]], 0.);
    assert_eq!(as_slice[[1, 3]], 10.);
    assert_eq!(as_slice[[1, 4]], 20.);

    // Third row: zeros in cols 0-2, then 30, 40 at cols 3-4
    assert_eq!(as_slice[[2, 0]], 0.);
    assert_eq!(as_slice[[2, 1]], 0.);
    assert_eq!(as_slice[[2, 2]], 0.);
    assert_eq!(as_slice[[2, 3]], 30.);
    assert_eq!(as_slice[[2, 4]], 40.);
}
