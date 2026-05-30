use fusor_gguf::GgmlType;
use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;

use crate::mir::inputs::MirValue;
use crate::mir::operation::Operation;
use crate::{
    CastTensor, DataType, DataTypeEnum, Device, Layout, LazyTensorData, Tensor, TensorData,
    TensorInfo,
    mir::{
        kernel_backend,
        kernel_backend::DirectKernel,
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
    nary_wise::UnaryFunctionChain,
};

use super::{QMatrix, QMatrixStorageLayout};

struct DequantizeDirectKernelVariant;

struct QDequantizeKernelParams {
    matrix_buffer: std::sync::Arc<wgpu::Buffer>,
    output_buffer: std::sync::Arc<wgpu::Buffer>,
    output_layout: tile_ir::Layout,
    output_element: tile_ir::ElementType,
    format: tile_ir::GgmlQuantFormat,
    k: u32,
    n: u32,
    dispatch_x: u32,
}

fn emit_qdequantize_kernel(
    kb: &mut tile_ir::KernelBuilder<std::sync::Arc<wgpu::Buffer>>,
    params: QDequantizeKernelParams,
) -> Option<()> {
    let q = tile_ir_kernels::quantized_matrix_for(
        kb,
        params.matrix_buffer,
        params.format,
        params.k,
        params.n,
    );
    let y = kb.write_element::<1>(
        params.output_element,
        tile_ir::KernelTensorRef::new(params.output_buffer, params.output_layout),
    );
    tile_ir_kernels::qdequantize(kb.program(), &q, &y, params.dispatch_x);
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
pub(crate) struct DequantizeOperation {
    pub(crate) matrix: QMatrix,
    pub(crate) datatype: DataTypeEnum,
    pub(crate) post_dequantize: UnaryFunctionChain,
}

impl DequantizeOperation {
    pub(crate) fn new(matrix: QMatrix, datatype: DataTypeEnum) -> Self {
        DequantizeOperation {
            matrix,
            datatype,
            post_dequantize: UnaryFunctionChain::empty(datatype),
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

impl Operation for DequantizeOperation {
    fn workgroup_shape_constraints(
        &self,
        _device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::Equals(16));
        constraints.add_constraint(1, Constraint::Equals(16));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
    ) -> [u32; 3] {
        let total = self
            .matrix
            .shape
            .iter()
            .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))
            .unwrap_or(u32::MAX);
        let lanes = workgroup_shape.x() * workgroup_shape.y() * workgroup_shape.z();
        [total.div_ceil(lanes), 1, 1]
    }

    fn visit_dependencies(&self, _: &mut dyn FnMut(crate::compute_graph::NodeIndex)) {}

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let shape = &self.matrix.shape;
        let output_tensor = TensorData::new_for_shape(&nodes.device(), shape, self.datatype);
        vec![self.matrix.clone().into(), output_tensor.into()]
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor = inputs[1].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        if !matches!(self.datatype, DataTypeEnum::F32 | DataTypeEnum::F16)
            || !self.post_dequantize.functions.is_empty()
            || (self.datatype == DataTypeEnum::F16 && !graph.device().f16_supported())
        {
            return None;
        }
        let [matrix, output] = inputs else {
            return None;
        };
        let MirValue::QMatrix(matrix) = matrix else {
            return None;
        };
        let output = output.as_tensor()?;
        if output.datatype() != self.datatype {
            return None;
        }

        let format = self.direct_quant_format()?;
        let k = *self.matrix.shape.last()? as u32;
        let n: u32 = self
            .matrix
            .shape
            .iter()
            .rev()
            .skip(1)
            .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))?;
        let total = k.checked_mul(n)?;
        let workgroups = total.div_ceil(256);
        let max_workgroups = graph
            .device()
            .limits()
            .max_compute_workgroups_per_dimension
            .max(1);
        let dispatch_x = workgroups.min(max_workgroups);
        let dispatch_y = workgroups.div_ceil(dispatch_x);
        if dispatch_y > max_workgroups {
            return None;
        }
        let cache_key = self.kernel_cache_key_with_dispatch(
            kernel_backend::KernelVariantKey::of::<DequantizeDirectKernelVariant>(),
            Some(_workgroup_shape),
            [dispatch_x, dispatch_y, 1],
            inputs,
        );
        let matrix_buffer = matrix.buffer().clone();
        let output_buffer = output.buffer().clone();
        let output_layout = tile_ir::Layout::contiguous(
            tile_ir::MemoryLevel::Storage,
            tile_ir::Shape::new([total]),
        );
        let output_datatype = self.datatype;
        kernel_backend::run_kernel(
            graph.device().kernel_cache(),
            self.name(),
            cache_key,
            [dispatch_x, dispatch_y, 1],
            move |kb| {
                emit_qdequantize_kernel(
                    kb,
                    QDequantizeKernelParams {
                        matrix_buffer,
                        output_buffer,
                        output_layout,
                        output_element: datatype_element(output_datatype)?,
                        format,
                        k,
                        n,
                        dispatch_x,
                    },
                )?;
                Some(())
            },
        )
    }

    fn name(&self) -> String {
        format!("dequantize_{}_to_{}", self.matrix.datatype, self.datatype)
    }
}

impl QMatrix {
    pub fn dequantize<T>(&self) -> Tensor
    where
        T: DataType,
        f32: CastTensor<T>,
    {
        if T::DATA_TYPE == DataTypeEnum::F16 && !self.device.f16_supported() {
            let tensor = self.dequantize::<f32>();
            return tensor.cast::<T>();
        }

        if matches!(self.datatype, GgmlType::F32 | GgmlType::F16) {
            let device = &self.device;
            let buffer = self.buffer.clone();
            let layout = Layout::contiguous(&self.shape);
            let datatype = match self.datatype {
                GgmlType::F32 => DataTypeEnum::F32,
                GgmlType::F16 => DataTypeEnum::F16,
                _ => unreachable!("dense matrix datatype checked above"),
            };
            let tensor = Tensor::from_parts(LazyTensorData::new(TensorData::new_from_parts(
                device, buffer, layout, datatype,
            )));
            return tensor.cast_to(T::DATA_TYPE);
        }

        let device = self.device.clone();
        let key = device
            .compute_graph()
            .dequantize(self.clone(), T::DATA_TYPE);

        let data = LazyTensorData::from_parts(
            device,
            TensorInfo::new(self.shape().into(), T::DATA_TYPE),
            key,
        );

        Tensor::from_parts(data)
    }
}
