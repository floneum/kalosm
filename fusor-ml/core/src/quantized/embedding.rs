use fusor_gguf::GgmlType;
use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;

use crate::{
    DataTypeEnum, Device, Tensor, TensorData, TensorInfo,
    compute_graph::NodeIndex,
    mir::{
        inputs::MirValue,
        kernel_backend,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    tensor::LazyTensorData,
    visit_tiled::distribute_workgroups,
};

use super::{QMatrix, QMatrixStorageLayout};

const BLOCK: usize = 256;

struct QEmbeddingDirectKernelVariant;

#[allow(clippy::too_many_arguments)]
fn emit_qembedding_kernel(
    kb: &mut tile_ir::KernelBuilder<std::sync::Arc<wgpu::Buffer>>,
    matrix_buffer: std::sync::Arc<wgpu::Buffer>,
    indexes_buffer: std::sync::Arc<wgpu::Buffer>,
    output_buffer: std::sync::Arc<wgpu::Buffer>,
    output_element: tile_ir::ElementType,
    format: tile_ir::GgmlQuantFormat,
    embedding_dim: u32,
    num_embeddings: u32,
    total: u32,
    dispatch_size: [u32; 3],
    indexes_offset: u32,
    indexes_layout: tile_ir::Layout,
    output_offset: u32,
    output_layout: tile_ir::Layout,
) -> Option<()> {
    let q = tile_ir_kernels::quantized_matrix_for(
        kb,
        matrix_buffer,
        format,
        embedding_dim,
        num_embeddings,
    );
    let indexes = kb.read::<tile_ir::U32, 2>(tile_ir::KernelTensorRef::with_offset(
        indexes_buffer,
        indexes_layout,
        indexes_offset,
    ));
    let y = kb.write_element::<2>(
        output_element,
        tile_ir::KernelTensorRef::with_offset(output_buffer, output_layout, output_offset),
    );

    kb.program()
        .program_grid::<BLOCK>(dispatch_size, |program| {
            let lane = program.lane();
            let group = program.program_id(tile_ir::WorkgroupAxis::X)
                + program.program_id(tile_ir::WorkgroupAxis::Y) * dispatch_size[0];
            let flat = group * BLOCK as u32 + lane;
            let in_bounds = flat.clone().lt(total);
            let dim = flat.clone() % embedding_dim;
            let index_pos = flat / embedding_dim;
            let token = program.load(
                indexes.at((0, index_pos.clone())),
                in_bounds.clone(),
                tile_ir::TileLiteral::U32(0),
            );
            let value = program.load_quantized(&q, dim.clone(), token, in_bounds.clone(), 0.0);
            program.store_element(y.at((index_pos, dim)), value, in_bounds);
        });
    Some(())
}

fn datatype_element(datatype: DataTypeEnum) -> Option<tile_ir::ElementType> {
    Some(match datatype {
        DataTypeEnum::F32 => tile_ir::ElementType::F32,
        DataTypeEnum::F16 => tile_ir::ElementType::F16,
        DataTypeEnum::U32 => return None,
    })
}

#[derive(Debug, Clone)]
pub(crate) struct QEmbeddingOperation {
    pub(crate) indexes: NodeIndex,
    pub(crate) matrix: QMatrix,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) datatype: DataTypeEnum,
}

impl QEmbeddingOperation {
    pub(crate) fn new(
        indexes: NodeIndex,
        index_count: usize,
        matrix: QMatrix,
        datatype: DataTypeEnum,
    ) -> Self {
        assert_eq!(
            matrix.shape.len(),
            2,
            "quantized embedding requires a 2D table, got {}D",
            matrix.shape.len()
        );
        let embedding_dim = matrix.shape[1];
        Self {
            indexes,
            matrix,
            out_shape: Box::new([index_count, embedding_dim]),
            datatype,
        }
    }

    fn direct_quant_format(&self) -> Option<tile_ir::GgmlQuantFormat> {
        Some(match self.matrix.datatype {
            GgmlType::Q4_0 if self.matrix.storage_layout() == QMatrixStorageLayout::Native => {
                tile_ir::GgmlQuantFormat::Q4_0Native
            }
            GgmlType::Q4_0 => tile_ir::GgmlQuantFormat::Q4_0,
            GgmlType::Q4_1 => tile_ir::GgmlQuantFormat::Q4_1,
            GgmlType::Q5_0 if self.matrix.storage_layout() == QMatrixStorageLayout::Native => {
                tile_ir::GgmlQuantFormat::Q5_0Native
            }
            GgmlType::Q5_0 => tile_ir::GgmlQuantFormat::Q5_0,
            GgmlType::Q5_1 => tile_ir::GgmlQuantFormat::Q5_1,
            GgmlType::Q8_0 if self.matrix.storage_layout() == QMatrixStorageLayout::Native => {
                tile_ir::GgmlQuantFormat::Q8_0Native
            }
            GgmlType::Q8_0 => tile_ir::GgmlQuantFormat::Q8_0,
            GgmlType::Q8_1 => tile_ir::GgmlQuantFormat::Q8_1,
            GgmlType::Q2K => tile_ir::GgmlQuantFormat::Q2K,
            GgmlType::Q3K => tile_ir::GgmlQuantFormat::Q3K,
            GgmlType::Q4K if self.matrix.storage_layout() == QMatrixStorageLayout::Native => {
                tile_ir::GgmlQuantFormat::Q4KNative
            }
            GgmlType::Q4K => tile_ir::GgmlQuantFormat::Q4K,
            GgmlType::Q5K if self.matrix.storage_layout() == QMatrixStorageLayout::Native => {
                tile_ir::GgmlQuantFormat::Q5KNative
            }
            GgmlType::Q5K => tile_ir::GgmlQuantFormat::Q5K,
            GgmlType::Q6K if self.matrix.storage_layout() == QMatrixStorageLayout::Native => {
                tile_ir::GgmlQuantFormat::Q6KNative
            }
            GgmlType::Q6K => tile_ir::GgmlQuantFormat::Q6K,
            GgmlType::Q8K => tile_ir::GgmlQuantFormat::Q8K,
            GgmlType::F16 | GgmlType::F32 => return None,
        })
    }
}

fn u32_layout_2d(layout: &crate::Layout) -> Option<(u32, tile_ir::Layout)> {
    let offset = layout.offset().try_into().ok()?;
    let shape = layout.shape();
    let strides = layout.strides();
    if shape.len() != 2 || strides.len() != 2 {
        return None;
    }
    Some((
        offset,
        tile_ir::Layout::strided(
            tile_ir::MemoryLevel::Storage,
            tile_ir::Shape::new([shape[0].try_into().ok()?, shape[1].try_into().ok()?]),
            &[strides[0].try_into().ok()?, strides[1].try_into().ok()?],
        ),
    ))
}

fn u32_index_layout(layout: &crate::Layout) -> Option<(u32, tile_ir::Layout)> {
    let offset = layout.offset().try_into().ok()?;
    let shape = layout.shape();
    let strides = layout.strides();
    if shape.len() != 1 || strides.len() != 1 {
        return None;
    }
    Some((
        offset,
        tile_ir::Layout::strided(
            tile_ir::MemoryLevel::Storage,
            tile_ir::Shape::new([1, shape[0].try_into().ok()?]),
            &[0, strides[0].try_into().ok()?],
        ),
    ))
}

impl Operation for QEmbeddingOperation {
    fn workgroup_shape_constraints(&self, _device: &Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::equals(BLOCK as u32));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(&self, _workgroup_shape: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3] {
        let total_elements: u64 = self.out_shape.iter().map(|&x| x as u64).product();
        let total_workgroups = total_elements.div_ceil(BLOCK as u64) as u32;
        let max_per_dim = inputs[2]
            .as_tensor()
            .unwrap()
            .device()
            .limits()
            .max_compute_workgroups_per_dimension;
        distribute_workgroups(total_workgroups, max_per_dim)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.indexes);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let indexes = nodes
            .get_result(self.indexes)
            .expect("QEmbedding indexes must be resolved before kernel launch");
        let device = nodes.device();
        let output = TensorData::new_for_shape(&device, &self.out_shape, self.datatype);
        vec![self.matrix.clone().into(), indexes.into(), output.into()]
    }

    fn output(
        &self,
        _nodes: &crate::compute_graph::ComputeGraphInner,
        inputs: &[MirValue],
    ) -> MirValue {
        inputs[2].as_tensor().unwrap().clone().into()
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let [matrix, indexes, output] = inputs else {
            return None;
        };
        let MirValue::QMatrix(matrix) = matrix else {
            return None;
        };
        let indexes = indexes.as_tensor()?;
        let output = output.as_tensor()?;
        if indexes.datatype() != DataTypeEnum::U32
            || output.datatype() != self.datatype
            || !matches!(self.datatype, DataTypeEnum::F32 | DataTypeEnum::F16)
            || (self.datatype == DataTypeEnum::F16 && !graph.device().f16_supported())
        {
            return None;
        }
        let [index_count, embedding_dim] = self.out_shape.as_ref() else {
            return None;
        };
        let [num_embeddings, matrix_embedding_dim] = self.matrix.shape.as_ref() else {
            return None;
        };
        if embedding_dim != matrix_embedding_dim {
            return None;
        }

        let format = self.direct_quant_format()?;
        let index_count: u32 = (*index_count).try_into().ok()?;
        let embedding_dim: u32 = (*embedding_dim).try_into().ok()?;
        let num_embeddings: u32 = (*num_embeddings).try_into().ok()?;
        let total = index_count.checked_mul(embedding_dim)?;
        let dispatch_size = self.dispatch_size(workgroup_shape, inputs);
        let (indexes_offset, indexes_layout) = u32_index_layout(indexes.layout())?;
        let (output_offset, output_layout) = u32_layout_2d(output.layout())?;
        let cache_key = self.kernel_cache_key_with_dispatch(
            kernel_backend::KernelVariantKey::of::<QEmbeddingDirectKernelVariant>(),
            Some(workgroup_shape),
            dispatch_size,
            inputs,
        );
        let matrix_buffer = matrix.buffer().clone();
        let indexes_buffer = indexes.buffer().clone();
        let output_buffer = output.buffer().clone();
        kernel_backend::run_kernel(
            graph.device().kernel_cache(),
            self.name(),
            cache_key,
            dispatch_size,
            move |kb| {
                let output_datatype = self.datatype;
                emit_qembedding_kernel(
                    kb,
                    matrix_buffer,
                    indexes_buffer,
                    output_buffer,
                    datatype_element(output_datatype)?,
                    format,
                    embedding_dim,
                    num_embeddings,
                    total,
                    dispatch_size,
                    indexes_offset,
                    indexes_layout,
                    output_offset,
                    output_layout,
                )?;
                Some(())
            },
        )
    }

    fn name(&self) -> String {
        format!(
            "q_embedding_{}_{}x{}",
            self.matrix.datatype, self.matrix.shape[0], self.matrix.shape[1]
        )
    }
}

impl QMatrix {
    pub fn index_select_rows(&self, indexes: &Tensor) -> Tensor {
        self.index_select_rows_to(indexes, DataTypeEnum::F32)
    }

    pub fn index_select_rows_to(&self, indexes: &Tensor, datatype: DataTypeEnum) -> Tensor {
        indexes.assert_rank::<1>();
        indexes.assert_datatype::<u32>();
        assert_eq!(
            self.shape.len(),
            2,
            "quantized row index_select requires a 2D table, got {}D",
            self.shape.len()
        );
        if matches!(self.datatype, GgmlType::F32 | GgmlType::F16) {
            let dense = if datatype == DataTypeEnum::F16 {
                self.dequantize::<half::f16>()
            } else {
                self.dequantize::<f32>()
            };
            return dense.index_select(0, indexes).cast_to(datatype);
        }
        let index_count = indexes.shape()[0];
        let device = self.device.clone();
        let datatype = if datatype == DataTypeEnum::F16 && !self.device.f16_supported() {
            DataTypeEnum::F32
        } else {
            datatype
        };
        let operation =
            QEmbeddingOperation::new(indexes.key(), index_count, self.clone(), datatype);
        let info = TensorInfo::new(operation.out_shape.clone(), datatype);
        let key = device.compute_graph().create_q_embedding(operation);
        Tensor::from_parts(LazyTensorData::from_parts(device, info, key))
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use fusor_gguf::BlockQ4K;

    use super::*;

    #[test]
    fn q4k_embedding_lookup_matches_dequantized_rows() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let shape = [4usize, BlockQ4K::BLOCK_SIZE];
            let block_count = shape.iter().product::<usize>() / BlockQ4K::BLOCK_SIZE;
            let raw_bytes = vec![0; block_count * size_of::<BlockQ4K>()];
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, shape.into(), fusor_gguf::GgmlType::Q4K)
                    .unwrap();
            let indexes = Tensor::new::<u32, 1, _>(&device, [0u32, 3u32].as_slice());

            let result = matrix
                .index_select_rows(&indexes)
                .as_slice::<2, f32>()
                .await
                .unwrap();

            assert_eq!(result.shape(), &[2, BlockQ4K::BLOCK_SIZE]);
            assert!(result.as_slice().iter().all(|value| *value == 0.0));
        });
    }
}
