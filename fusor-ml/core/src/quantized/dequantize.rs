use fusor_gguf::GgmlType;

use crate::Layout;
use crate::mir::inputs::MirValue;
use crate::mir::operation::Operation;
use crate::mir::workgroup_shape::WorkgroupShapeConstraints;
use crate::{
    DataType, DataTypeEnum, Device, ElementWiseFunctions, LazyTensorData, Tensor, TensorData,
    TensorInfo, mir::kernel::GenericKernel,
};
use std::fmt::Write;

use super::{QMatrix, dequantize_block};

#[derive(Debug, Clone)]
pub(crate) struct DequantizeOperation {
    pub(crate) matrix: QMatrix,
    pub(crate) datatype: DataTypeEnum,
    pub(crate) post_dequantize: ElementWiseFunctions,
}

impl DequantizeOperation {
    pub(crate) fn new(matrix: QMatrix, datatype: DataTypeEnum) -> Self {
        DequantizeOperation {
            matrix,
            datatype,
            post_dequantize: ElementWiseFunctions::empty(datatype),
        }
    }

    fn elements_per_block(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }
}

impl Operation for DequantizeOperation {
    fn workgroup_shape_constraints(
        &self,
        _: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        WorkgroupShapeConstraints::new()
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
    ) -> [u32; 3] {
        // Linearize dispatch for high-dimensional tensors
        let elements_per_block = self.elements_per_block();
        let total_blocks: u32 = self
            .matrix
            .shape
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                if i == self.matrix.shape.len() - 1 {
                    (n as u32).div_ceil(elements_per_block)
                } else {
                    n as u32
                }
            })
            .product();

        let workgroup_volume = workgroup_shape.x() * workgroup_shape.y() * workgroup_shape.z();
        let total_workgroups = total_blocks.div_ceil(workgroup_volume);

        // Distribute workgroups across x, y, z dimensions
        let max_per_dim = self
            .matrix
            .device
            .limits()
            .max_compute_workgroups_per_dimension;
        let workgroup_size_x = total_workgroups.min(max_per_dim);
        let remaining = total_workgroups.div_ceil(workgroup_size_x);
        let workgroup_size_y = remaining.min(max_per_dim);
        let workgroup_size_z = total_workgroups.div_ceil(workgroup_size_x * workgroup_size_y);

        [workgroup_size_x, workgroup_size_y, workgroup_size_z]
    }

    fn visit_dependencies(&self, _: &mut dyn FnMut(crate::compute_graph::NodeIndex)) {}

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let shape = &self.matrix.shape;
        let datatype = self.datatype;
        let output_tensor = TensorData::new_for_shape(&nodes.device(), shape, datatype);
        vec![MirValue::from(self.matrix.clone()), output_tensor.into()]
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor = inputs[1].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn build_kernel(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        let datatype = self.datatype;
        let rank = self.matrix.shape.len() as u32;

        let input = kernel.add_q_matrix_input(rank, self.matrix.datatype);
        let output = kernel.add_tensor_input(rank, true, datatype);

        let post_element_wise = self.post_dequantize.add_functions(kernel);
        let process_output = |input: &str| {
            post_element_wise
                .iter()
                .fold(input.to_string(), |acc, f| f.call(vec![acc]))
        };

        let elements_per_block = self.elements_per_block();

        // Linearize the global index to handle more than 3D inputs
        let linearized_workgroup_index = workgroup_shape.linearized_workgroup_index(kernel);
        let local_id = kernel.workgroup_local_index();
        writeln!(
            kernel,
            "let flat_global_id = ({linearized_workgroup_index}) * BLOCKSIZE + {local_id};"
        )
        .unwrap();

        // Convert flat index to multi-dimensional indices (row-major order)
        writeln!(kernel, "var remaining_index = flat_global_id;").unwrap();
        for dim in (0..rank).rev() {
            let shape_binding = output.shape_binding(dim);
            let divisor = if dim == rank - 1 {
                format!("(({shape_binding} + {elements_per_block} - 1u) / {elements_per_block}u)")
            } else {
                format!("{shape_binding}")
            };
            writeln!(kernel, "let index_{dim} = remaining_index % {divisor};").unwrap();
            if dim > 0 {
                writeln!(kernel, "remaining_index = remaining_index / {divisor};").unwrap();
            }
        }

        write!(kernel, "let chunk = {input}[").unwrap();
        input.strided_index(kernel, (0..).map(|i| format!("index_{i}")));
        writeln!(kernel, "];").unwrap();

        dequantize_block(
            kernel,
            self.matrix.datatype,
            "chunk".to_string(),
            datatype,
            |i, data, kernel| {
                let indexes: Box<[_]> = (0..rank)
                    .map(|dim| {
                        let base = format!("index_{dim}");
                        if dim == rank - 1 {
                            format!("{base} * {elements_per_block} + {i}")
                        } else {
                            base
                        }
                    })
                    .collect();
                output.check_bounds(kernel, indexes.clone(), |kernel| {
                    write!(kernel, "let output_index = ").unwrap();
                    output.strided_index(kernel, indexes);
                    writeln!(kernel, ";").unwrap();

                    writeln!(
                        kernel,
                        "{output}[output_index] = {};",
                        process_output(&data)
                    )
                    .unwrap();
                });
            },
        );
    }

    fn name(&self) -> String {
        format!("dequantize_{}_to_{}", self.matrix.datatype, self.datatype)
    }
}

impl QMatrix {
    pub fn dequantize<const R: usize, T: DataType>(&self) -> Tensor<R, T> {
        assert_eq!(
            self.shape.len(),
            R,
            "Dequantize: expected {}D tensor, got {}D tensor. Shape: {:?}",
            R,
            self.shape.len(),
            self.shape
        );

        // If the types already match, just return a view of the existing data
        // Note: Only use f16 directly if the device supports it
        if self.datatype == GgmlType::F32 && T::WGSL_TYPE == DataTypeEnum::F32
            || self.datatype == GgmlType::F16
                && T::WGSL_TYPE == DataTypeEnum::F16
                && self.device.f16_supported()
        {
            let device = &self.device;
            let buffer = self.buffer.clone();
            let layout = Layout::contiguous(&self.shape);
            let datatype = T::WGSL_TYPE;
            return Tensor::from_parts(LazyTensorData::new(TensorData::new_from_parts(
                device, buffer, layout, datatype,
            )));
        }

        let device = self.device.clone();
        let key = device
            .compute_graph()
            .dequantize(self.clone(), T::WGSL_TYPE);

        let data = LazyTensorData::from_parts(
            device,
            TensorInfo::new(self.shape().into(), T::WGSL_TYPE),
            key,
        );

        Tensor::from_parts(data)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_dequantize_smol_lm() {
    use crate::Device;
    use fusor_gguf::GgufMetadata;

    let device = Device::test_instance();

    let url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";
    let bytes = reqwest::get(url).await.unwrap().bytes().await.unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let metadata = GgufMetadata::read(&mut reader).unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let candle_metadata = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();

    for (name, candle_q_matrix_metadata) in candle_metadata.tensor_infos {
        let tensor = metadata.tensor_infos.get(&*name).unwrap();
        println!("{name}: {tensor:?}");

        let candle_q_tensor = candle_q_matrix_metadata
            .read(
                &mut reader,
                candle_metadata.tensor_data_offset,
                &candle_core::Device::Cpu,
            )
            .unwrap();
        let candle_result = candle_q_tensor
            .dequantize(&candle_core::Device::Cpu)
            .unwrap();

        let candle_result_doubled = (&candle_result * 2.0).unwrap();

        let q_matrix_metadata = metadata.tensor_infos.get(&*name).unwrap();

        let q_matrix = QMatrix::read(
            &device,
            q_matrix_metadata,
            &mut reader,
            metadata.tensor_data_offset,
        )
        .unwrap();
        assert_eq!(candle_result.shape().dims(), q_matrix.shape());

        match candle_result.rank() {
            1 => {
                let fusor_result = q_matrix.dequantize::<1, f32>();
                let candle_result = candle_result.to_vec1::<f32>().unwrap();
                let result = fusor_result.as_slice().await.unwrap();
                for i in 0..candle_result.len() {
                    let expected = candle_result[i];
                    let actual = result[[i]];
                    if (expected - actual).abs() > 0.01 {
                        assert_eq!(
                            expected, actual,
                            "Mismatch at ({i}) - expected: {expected}, actual: {actual}"
                        );
                    }
                }

                let fusor_result = q_matrix.dequantize::<1, f32>() * 2.0;
                let candle_result = candle_result_doubled.to_vec1::<f32>().unwrap();
                let result = fusor_result.as_slice().await.unwrap();
                for i in 0..candle_result.len() {
                    let expected = candle_result[i];
                    let actual = result[[i]];
                    if (expected - actual).abs() > 0.01 {
                        assert_eq!(
                            expected, actual,
                            "Mismatch at ({i}) - expected: {expected}, actual: {actual}"
                        );
                    }
                }
            }
            2 => {
                let fusor_result = q_matrix.dequantize::<2, f32>();
                let candle_result = candle_result.to_vec2::<f32>().unwrap();
                let result = fusor_result.as_slice().await.unwrap();
                for x in 0..candle_result.len() {
                    for y in 0..candle_result[0].len() {
                        let expected = candle_result[x][y];
                        let actual = result[[x, y]];
                        if (expected - actual).abs() > 0.01 {
                            assert_eq!(
                                expected, actual,
                                "Mismatch at ({x}, {y}) - expected: {expected}, actual: {actual}"
                            );
                        }
                    }
                }

                let fusor_result = q_matrix.dequantize::<2, f32>() * 2.0;
                let candle_result = candle_result_doubled.to_vec2::<f32>().unwrap();
                let result = fusor_result.as_slice().await.unwrap();
                for x in 0..candle_result.len() {
                    for y in 0..candle_result[0].len() {
                        let expected = candle_result[x][y];
                        let actual = result[[x, y]];
                        if (expected - actual).abs() > 0.01 {
                            assert_eq!(
                                expected, actual,
                                "Mismatch at ({x}, {y}) - expected: {expected}, actual: {actual}"
                            );
                        }
                    }
                }
            }
            _ => todo!(),
        }
    }
}
