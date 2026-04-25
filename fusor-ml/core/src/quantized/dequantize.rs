use fusor_gguf::GgmlType;

use crate::Layout;
use crate::mir::inputs::MirValue;
use crate::mir::operation::Operation;
use crate::{DataType, DataTypeEnum, LazyTensorData, Tensor, TensorData, TensorInfo};

use super::QMatrix;

#[derive(Debug, Clone)]
pub(crate) struct DequantizeOperation {
    pub(crate) matrix: QMatrix,
    pub(crate) datatype: DataTypeEnum,
}

impl DequantizeOperation {
    pub(crate) fn new(matrix: QMatrix, datatype: DataTypeEnum) -> Self {
        DequantizeOperation { matrix, datatype }
    }
}

impl Operation for DequantizeOperation {
    fn visit_dependencies(&self, _: &mut dyn FnMut(crate::compute_graph::NodeIndex)) {}

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let shape = &self.matrix.shape;
        let datatype = self.datatype;
        let output_tensor = TensorData::new_for_shape(&nodes.device(), shape, datatype);
        vec![MirValue::from(self.matrix.clone()), output_tensor.into()]
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
