use std::hash::Hash;

use fusor_gguf::GgmlType;
use fusor_tile_ir as tile_ir;

use crate::compute_graph::NodeIndex;
use crate::mir::inputs::MirValue;
use crate::mir::operation::Operation;
use crate::{
    CastTensor, DataType, DataTypeEnum, Device, Layout, LazyTensorData, Tensor, TensorData,
    TensorInfo,
    mir::{
        kernel_backend::DirectKernel,
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
    nary_wise::{ElementwiseOperation, NaryExpr, NaryFunction, UnaryFunctionChain},
};

use super::{QMatrix, QMatrixStorageLayout};

#[derive(Debug, Clone, PartialEq)]
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
}

/// The tile-ir quantization format for a matrix, accounting for its storage
/// layout. `None` for dense f16/f32 storage.
pub(crate) fn quant_format(matrix: &QMatrix) -> Option<tile_ir::GgmlQuantFormat> {
    Some(match matrix.datatype {
        GgmlType::Q4_0 if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q4_0Native
        }
        GgmlType::Q4_0 => tile_ir::GgmlQuantFormat::Q4_0,
        GgmlType::Q4_1 => tile_ir::GgmlQuantFormat::Q4_1,
        GgmlType::Q5_0 if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q5_0Native
        }
        GgmlType::Q5_0 => tile_ir::GgmlQuantFormat::Q5_0,
        GgmlType::Q5_1 => tile_ir::GgmlQuantFormat::Q5_1,
        GgmlType::Q8_0 if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q8_0Native
        }
        GgmlType::Q8_0 => tile_ir::GgmlQuantFormat::Q8_0,
        GgmlType::Q8_1 => tile_ir::GgmlQuantFormat::Q8_1,
        GgmlType::Q2K => tile_ir::GgmlQuantFormat::Q2K,
        GgmlType::Q3K => tile_ir::GgmlQuantFormat::Q3K,
        GgmlType::Q4K if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q4KNative
        }
        GgmlType::Q4K => tile_ir::GgmlQuantFormat::Q4K,
        GgmlType::Q5K if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q5KNative
        }
        GgmlType::Q5K => tile_ir::GgmlQuantFormat::Q5K,
        GgmlType::Q6K if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q6KNative
        }
        GgmlType::Q6K => tile_ir::GgmlQuantFormat::Q6K,
        GgmlType::Q8K => tile_ir::GgmlQuantFormat::Q8K,
        GgmlType::F16 | GgmlType::F32 => return None,
    })
}

impl Operation for DequantizeOperation {
    fn hash_kernel_fields(&self, state: &mut rustc_hash::FxHasher) {
        self.matrix.datatype().hash(state);
        self.matrix.storage_layout().hash(state);
        self.matrix.shape().hash(state);
        self.datatype.hash(state);
        self.post_dequantize.hash(state);
    }

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

    fn visit_dependencies_mut(&mut self, _: &mut dyn FnMut(&mut crate::compute_graph::NodeIndex)) {}

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
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        // Dequantization is the identity expression over one block-quantized
        // input: the generic n-ary kernel decodes per element through the
        // format-aware load, and any fused chain rides along.
        let rank = self.matrix.shape.len();
        let mut expression = NaryExpr::input(0, rank);
        let mut current = DataTypeEnum::F32;
        if self.post_dequantize.input_datatype() != current
            || self.post_dequantize.out_datatype() != self.datatype
        {
            expression = NaryExpr::Op {
                children: vec![expression],
                function: NaryFunction::unary(
                    Some("cast".to_string()),
                    crate::nary_wise::NaryOp::Cast,
                    current,
                    self.post_dequantize.input_datatype(),
                ),
            };
            current = self.post_dequantize.input_datatype();
        }
        for function in &self.post_dequantize.functions {
            expression = NaryExpr::Op {
                children: vec![expression],
                function: function.clone(),
            };
            current = function.output_type;
        }
        if current != self.datatype {
            expression = NaryExpr::Op {
                children: vec![expression],
                function: NaryFunction::unary(
                    Some("cast".to_string()),
                    crate::nary_wise::NaryOp::Cast,
                    current,
                    self.datatype,
                ),
            };
        }
        let synthesized = ElementwiseOperation {
            inputs: vec![NodeIndex::new(0)],
            expression,
            shape: self.matrix.shape.clone(),
            output_datatype: self.datatype,
        };
        crate::nary_direct::build_nary_direct_kernel(&synthesized, graph, workgroup_shape, inputs)
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
