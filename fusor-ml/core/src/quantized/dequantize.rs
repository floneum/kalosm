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

use super::QMatrix;

struct DequantizeDirectKernelVariant;

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
        if self.datatype != DataTypeEnum::F32 || !self.post_dequantize.functions.is_empty() {
            return None;
        }
        let [matrix, output] = inputs else {
            return None;
        };
        let MirValue::QMatrix(matrix) = matrix else {
            return None;
        };
        let output = output.as_tensor()?;
        if output.datatype() != DataTypeEnum::F32 {
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
        kernel_backend::run_kernel(
            graph.device().kernel_cache(),
            self.name(),
            cache_key,
            [dispatch_x, dispatch_y, 1],
            move |kb| {
                let q = tile_ir_kernels::quantized_matrix_for(kb, matrix_buffer, format, k, n);
                let y = kb.write::<tile_ir::F32, 1>(tile_ir::KernelTensorRef::new(
                    output_buffer,
                    output_layout,
                ));
                tile_ir_kernels::qdequantize(kb.program(), &q, &y, dispatch_x);
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
        if T::DATA_TYPE != DataTypeEnum::F32 {
            let tensor = self.dequantize::<f32>();
            return tensor.cast::<T>();
        }

        // If the types already match, just return a view of the existing data
        if self.datatype == GgmlType::F32 {
            let device = &self.device;
            let buffer = self.buffer.clone();
            let layout = Layout::contiguous(&self.shape);
            let datatype = T::DATA_TYPE;
            return Tensor::from_parts(LazyTensorData::new(TensorData::new_from_parts(
                device, buffer, layout, datatype,
            )));
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
