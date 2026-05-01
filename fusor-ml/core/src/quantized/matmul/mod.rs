use crate::{
    DataTypeEnum, Device, Layout, Tensor, TensorData,
    compute_graph::NodeIndex,
    matmul::MatMulOperation,
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding},
        inputs::MirValue,
        operation::Operation,
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout,
            tile_storage_write_with_direct_layout,
        },
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
};
use fusor_gguf::GgmlType;
use phase_token_prototype as tile_ir;

use super::QMatrix;

#[derive(Debug, Clone)]
pub(crate) struct QMatMulOperation {
    pub(crate) input_datatype: DataTypeEnum,
    pub(crate) input: NodeIndex,
    pub(crate) matrix: QMatrix,
    pub(crate) in_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
}

impl QMatMulOperation {
    pub(crate) fn new(
        input_datatype: DataTypeEnum,
        input_shape: &[usize],
        input: NodeIndex,
        matrix: QMatrix,
    ) -> Self {
        let last_dim = input_shape.len() - 1;
        let mut out_shape = input_shape.to_vec();
        out_shape[last_dim] = matrix.shape[0];
        assert_eq!(input_shape[last_dim], matrix.shape[1]);
        let out_shape = out_shape.into_boxed_slice();
        QMatMulOperation {
            input_datatype,
            input,
            matrix,
            in_shape: input_shape.into(),
            out_shape,
        }
    }

    fn m_size(&self) -> u32 {
        let m_dim_idx = self.in_shape.len() - 2;
        self.in_shape[m_dim_idx] as u32
    }

    fn n_size(&self) -> u32 {
        self.matrix.shape[0] as u32
    }

    fn direct_quant_format(&self) -> Option<tile_ir::GgmlQuantFormat> {
        Some(match self.matrix.datatype {
            GgmlType::Q4_0 => tile_ir::GgmlQuantFormat::Q4_0,
            GgmlType::Q4_1 => tile_ir::GgmlQuantFormat::Q4_1,
            GgmlType::Q5_0 => tile_ir::GgmlQuantFormat::Q5_0,
            GgmlType::Q5_1 => tile_ir::GgmlQuantFormat::Q5_1,
            GgmlType::Q8_0 => tile_ir::GgmlQuantFormat::Q8_0,
            GgmlType::Q8_1 => tile_ir::GgmlQuantFormat::Q8_1,
            GgmlType::Q2K => tile_ir::GgmlQuantFormat::Q2K,
            GgmlType::Q3K => tile_ir::GgmlQuantFormat::Q3K,
            GgmlType::Q4K => tile_ir::GgmlQuantFormat::Q4K,
            GgmlType::Q5K => tile_ir::GgmlQuantFormat::Q5K,
            GgmlType::Q6K => tile_ir::GgmlQuantFormat::Q6K,
            GgmlType::Q8K => tile_ir::GgmlQuantFormat::Q8K,
            GgmlType::F16 | GgmlType::F32 => return None,
        })
    }
}

fn ceil_div_u32(x: u32, divisor: u32) -> u32 {
    x.div_ceil(divisor)
}

fn split_workgroups_2d(
    total_workgroups: u32,
    max_workgroups_per_dimension: u32,
) -> Option<[u32; 2]> {
    if total_workgroups == 0 {
        return Some([1, 1]);
    }

    let max_workgroups_per_dimension = max_workgroups_per_dimension.max(1);
    let x = total_workgroups.min(max_workgroups_per_dimension);
    let y = ceil_div_u32(total_workgroups, x);
    (y <= max_workgroups_per_dimension).then_some([x, y])
}

impl<const R: usize> Tensor<R, f32> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        self.add_q_mat_mul(other)
    }
}

impl<const R: usize> Tensor<R, half::f16> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        self.cast::<f32>().q_mat_mul(other).cast()
    }
}

impl Operation for QMatMulOperation {
    fn workgroup_shape_constraints(
        &self,
        _device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        if self.m_size() == 1 {
            constraints.add_constraint(0, Constraint::Equals(1));
        } else {
            constraints.add_constraint(0, Constraint::Equals(32));
        }
        constraints.add_constraint(1, Constraint::Equals(1));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
    ) -> [u32; 3] {
        let n = self.n_size();
        let m = self.m_size();
        // Calculate batch size for dimensions beyond the last two (M, K)
        let batch_size: u32 = self
            .in_shape
            .iter()
            .rev()
            .skip(2)
            .map(|x| *x as u32)
            .product();

        if m == 1 {
            [n, 1, batch_size]
        } else {
            [n, m, batch_size]
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_result(self.input).unwrap();
        let q_matrix = self.matrix.clone();
        let device = input.device();
        let output_tensor = TensorData::new_for_shape(device, &self.out_shape, input.datatype());
        vec![input.into(), q_matrix.into(), output_tensor.into()]
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let [input, matrix, output] = inputs else {
            return None;
        };
        let input = input.as_tensor()?;
        let MirValue::QMatrix(matrix) = matrix else {
            return None;
        };
        let output = output.as_tensor()?;
        if input.datatype() != DataTypeEnum::F32 || output.datatype() != DataTypeEnum::F32 {
            return None;
        }
        if matrix.datatype() == GgmlType::F32 {
            return self.build_dense_direct_kernel(graph, input, matrix, output);
        }
        let input_rank = input.layout().shape().len();
        if input_rank != output.layout().shape().len() {
            return None;
        }

        let format = self.direct_quant_format()?;
        let a_view = flatten_matrix_layout(input.layout())?;
        let y_view = flatten_matrix_layout(output.layout())?;
        let m = a_view.rows;
        let k = a_view.cols;
        let y_m = y_view.rows;
        let n = y_view.cols;
        if m != y_m || k != self.matrix.shape[1] as u32 || n != self.matrix.shape[0] as u32 {
            return None;
        }

        let mut qmatmul_workgroups_x = 1;
        if m == 1 {
            let qgemv_workgroups = n.div_ceil(format.qgemv_cols_per_workgroup());
            let [dispatch_x, _] = split_workgroups_2d(
                qgemv_workgroups,
                graph.device().limits().max_compute_workgroups_per_dimension,
            )?;
            qmatmul_workgroups_x = dispatch_x;
        }
        let ir = tile_ir::tile::build(move |phase| {
            let a = tile_storage_read_with_direct_layout(phase, a_view);
            let b = phase.quantized_matrix(format, k, n);
            let y = tile_storage_write_with_direct_layout(phase, y_view);
            if m == 1 {
                phase.qgemv::<4, 64>(&a, &b, &y, 4, qmatmul_workgroups_x);
            } else {
                phase.qmatmul::<8, 4, 8>(&a, &b, &y, 4);
            }
        });
        let dispatch_size = ir.single_tile_program_grid()?;
        let max_workgroups = graph.device().limits().max_compute_workgroups_per_dimension;
        if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
            return None;
        }
        let cache_key = format!(
            "{}:direct:{format:?}:m={m}:k={k}:n={n}:dispatch={dispatch_size:?}:{:?}:{:?}",
            self.name(),
            input.layout(),
            output.layout()
        );
        let module =
            if let Some(module) = graph.device().naga_module_cache().write().get(&cache_key) {
                module.clone()
            } else {
                let module = ir.lower_to_naga().ok()?.module().clone();
                graph
                    .device()
                    .naga_module_cache()
                    .write()
                    .get_or_insert(cache_key.clone(), || module.clone())
                    .clone()
            };

        Some(DirectKernel::new_with_cache_key(
            self.name(),
            cache_key,
            module,
            vec![
                DirectKernelBinding::Storage {
                    binding: 0,
                    buffer: input.buffer().clone(),
                    read_only: true,
                },
                DirectKernelBinding::Storage {
                    binding: 1,
                    buffer: matrix.buffer().clone(),
                    read_only: true,
                },
                DirectKernelBinding::Storage {
                    binding: 2,
                    buffer: output.buffer().clone(),
                    read_only: false,
                },
            ],
            dispatch_size,
        ))
    }

    fn requires_single_kernel_batch(&self) -> bool {
        true
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor = inputs[2].as_tensor().unwrap();
        output_tensor.clone().into()
    }

    fn name(&self) -> String {
        format!(
            "q_mat_mul_{}_{}_{}_{}",
            self.input_datatype,
            self.in_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.matrix.datatype,
            self.matrix
                .shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

impl QMatMulOperation {
    fn build_dense_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        input: &TensorData,
        matrix: &QMatrix,
        output: &TensorData,
    ) -> Option<DirectKernel> {
        let [n, k] = matrix.shape() else {
            return None;
        };
        let (n, k) = (*n, *k);
        let input_shape = input.layout().shape();
        let rank = input_shape.len();
        if rank < 2 {
            return None;
        }
        let mut dense_shape = input_shape.to_vec();
        dense_shape[rank - 2] = k;
        dense_shape[rank - 1] = n;
        let mut dense_strides = vec![0; rank];
        dense_strides[rank - 2] = 1;
        dense_strides[rank - 1] = k;
        let dense_weight_t = TensorData::new_from_parts(
            matrix.device(),
            matrix.buffer().clone(),
            Layout::from_parts(
                0,
                dense_shape.into_boxed_slice(),
                dense_strides.into_boxed_slice(),
            ),
            DataTypeEnum::F32,
        );
        let device = graph.device();
        let dense_matmul = MatMulOperation::new(
            DataTypeEnum::F32,
            self.input,
            self.input,
            input.layout().shape(),
            dense_weight_t.layout().shape(),
            None,
            &device,
        );
        dense_matmul.build_direct_kernel(
            graph,
            &dense_matmul
                .workgroup_shape_constraints(&device)
                .solve(device.max_subgroup_size())?,
            &[
                input.clone().into(),
                dense_weight_t.into(),
                output.clone().into(),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{mem::size_of, sync::Arc};

    use fusor_gguf::{BlockQ4_0, BlockQ8_0, GgufBlock};

    use super::*;
    use crate::{
        compute_graph::{ComputeGraphInner, ComputeGraphNodes},
        mir::workgroup_shape::WorkgroupShape,
    };

    fn padded_copy_size(size: u64) -> u64 {
        let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
        ((size + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT)
    }

    #[tokio::test]
    async fn qmatmul_direct_kernel_binds_compact_quantized_weight_buffer() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let weight_shape = [128usize, 256usize];
        let element_count = weight_shape.iter().product::<usize>();
        let block_count = element_count / BlockQ4_0::BLOCK_SIZE;
        let raw_bytes = vec![0; block_count * size_of::<BlockQ4_0>()];
        let matrix =
            QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4_0).unwrap();

        let compact_len = block_count * size_of::<<BlockQ4_0 as GgufBlock>::BytesF32>();
        let dense_len = element_count * size_of::<f32>();
        assert_eq!(matrix.buffer().size(), padded_copy_size(compact_len as u64));
        assert!(matrix.buffer().size() < padded_copy_size(dense_len as u64));

        let input = TensorData::new_for_shape(&device, &[1, weight_shape[1]], DataTypeEnum::F32);
        let output = TensorData::new_for_shape(&device, &[1, weight_shape[0]], DataTypeEnum::F32);
        let graph = ComputeGraphInner {
            device: device.downgrade(),
            nodes: ComputeGraphNodes::default(),
        };
        let operation = QMatMulOperation {
            input_datatype: DataTypeEnum::F32,
            input: NodeIndex::new(0),
            matrix: matrix.clone(),
            in_shape: Box::new([1, weight_shape[1]]),
            out_shape: Box::new([1, weight_shape[0]]),
        };
        let kernel = operation
            .build_direct_kernel(
                &graph,
                &WorkgroupShape::new(256, 1, 1),
                &[input.into(), matrix.clone().into(), output.into()],
            )
            .expect("qmatmul should build a direct quantized kernel");

        let bindings = kernel.bindings_for_test();
        assert_eq!(bindings.len(), 3);
        let DirectKernelBinding::Storage {
            binding,
            buffer,
            read_only,
        } = &bindings[1];
        assert_eq!(*binding, 1);
        assert!(*read_only);
        assert!(Arc::ptr_eq(buffer, matrix.buffer()));
    }

    #[tokio::test]
    async fn qmatmul_accepts_dense_f32_qmatrix_without_generic_fallback() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let matrix = QMatrix::from_parts(
            &device,
            bytemuck::cast_slice(&weights),
            Box::new([2usize, 4usize]),
            GgmlType::F32,
        )
        .unwrap();
        let input_rows = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let input: Tensor<2, f32> = Tensor::new(&device, &input_rows);

        let result = input.q_mat_mul(&matrix).as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2]);
        assert!((result[[0, 0]] - 30.0).abs() < 1e-4);
        assert!((result[[0, 1]] - 70.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn q5_0_qgemv_matches_expected_values() {
        let Ok(device) = Device::new().await else {
            return;
        };

        fn q5_0_block(scale: f32, high_bits: [u8; 4], low_bits: u8) -> Vec<u8> {
            let mut bytes = Vec::with_capacity(22);
            bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend_from_slice(&high_bits);
            bytes.extend(std::iter::repeat_n(low_bits, 16));
            bytes
        }

        let mut raw_bytes = Vec::new();
        raw_bytes.extend(q5_0_block(1.0, [0xff; 4], 0x11));
        raw_bytes.extend(q5_0_block(1.0, [0x00; 4], 0xff));
        let matrix =
            QMatrix::from_parts(&device, &raw_bytes, Box::new([2, 32]), GgmlType::Q5_0).unwrap();
        let input_rows = vec![(1..=32).map(|value| value as f32).collect::<Vec<_>>()];
        let input: Tensor<2, f32> = Tensor::new(&device, &input_rows);

        let result = input.q_mat_mul(&matrix).as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2]);
        assert!((result[[0, 0]] - 528.0).abs() < 1e-3);
        assert!((result[[0, 1]] + 528.0).abs() < 1e-3);
    }

    #[tokio::test]
    async fn f16_qmatmul_casts_through_f32_direct_path() {
        let Ok(device) = Device::new().await else {
            return;
        };
        if !device.f16_supported() {
            return;
        }

        let weight_shape = [4usize, BlockQ8_0::BLOCK_SIZE];
        let block_count = weight_shape.iter().product::<usize>() / BlockQ8_0::BLOCK_SIZE;
        let raw_bytes = vec![0; block_count * size_of::<BlockQ8_0>()];
        let matrix =
            QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q8_0).unwrap();
        let input_rows = vec![vec![half::f16::from_f32(0.25); weight_shape[1]]];
        let input: Tensor<2, half::f16> = Tensor::new(&device, &input_rows);

        let result = input.q_mat_mul(&matrix).as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, weight_shape[0]]);
        assert!(
            result
                .as_slice()
                .iter()
                .take(weight_shape[0])
                .all(|value| *value == half::f16::from_f32(0.0))
        );
    }
}
