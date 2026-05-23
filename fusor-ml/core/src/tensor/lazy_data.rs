#[cfg(feature = "graphvis")]
use tabbycat::Graph;

use crate::{
    Device, FlashAttentionOperation, Layout, MatMulOperation, ReduceOperation,
    compute_graph::NodeIndex,
    map_layout::MapLayoutOperation,
    nary_wise::{NaryExpr, NaryFunction, NaryOperation},
    quantized::matmul::QMatMulOperation,
    resize::ResizeOperation,
    rms_norm::RmsNormOperation,
    slice_assign::SliceAssignOperation,
};

use super::{TensorData, TensorInfo};

pub(crate) struct LazyTensorData {
    pub(crate) device: Device,
    pub(crate) info: TensorInfo,
    pub(crate) key: NodeIndex,
}

impl Clone for LazyTensorData {
    fn clone(&self) -> Self {
        self.device.compute_graph().add_reference(self.key);
        Self {
            device: self.device.clone(),
            info: self.info.clone(),
            key: self.key,
        }
    }
}

impl Drop for LazyTensorData {
    fn drop(&mut self) {
        self.device.compute_graph().remove_reference(self.key);
    }
}

impl LazyTensorData {
    pub(crate) fn new(data: TensorData) -> Self {
        let device = data.device.clone();
        let info = data.info.clone();
        let key = device.compute_graph().create_tensor(data);

        Self {
            device,
            info: TensorInfo::new(info.shape().into(), info.datatype()),
            key,
        }
    }

    pub(crate) fn from_parts(device: Device, info: TensorInfo, key: NodeIndex) -> Self {
        Self { device, info, key }
    }

    pub(crate) fn nary(&self, nary: NaryOperation) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let key = device.compute_graph().create_nary(nary);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn unary_nary(&self, function: NaryFunction) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.datatype = function.output_type;
        let rank = info.rank();
        let nary = NaryOperation {
            inputs: vec![self.key],
            expression: NaryExpr::Op {
                children: vec![NaryExpr::input(0, rank)],
                function,
            },
            shape: info.shape().into(),
            output_datatype: info.datatype,
        };
        let key = device.compute_graph().create_nary(nary);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn binary_nary(
        &self,
        other_key: NodeIndex,
        function: NaryFunction,
        shape: &[usize],
    ) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.datatype = function.output_type;
        let rank = shape.len();
        let nary = NaryOperation {
            inputs: vec![self.key, other_key],
            expression: NaryExpr::Op {
                children: vec![NaryExpr::input(0, rank), NaryExpr::input(1, rank)],
                function,
            },
            shape: shape.into(),
            output_datatype: info.datatype,
        };
        let key = device.compute_graph().create_nary(nary);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn mat_mul(&self, function: MatMulOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = function.out_shape.clone();
        let key = device.compute_graph().create_mat_mul(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn q_mat_mul(&self, function: QMatMulOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = function.out_shape.clone();
        let key = device.compute_graph().create_q_mat_mul(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn reduce(&self, function: ReduceOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        let dim = function.axis;
        let input_shape = self.info.shape();
        let new_shape: Box<[usize]> = input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect();
        // Short-circuit empty inputs: any zero-sized dim makes the input
        // empty. Reducing along the zero axis yields the reduction's identity
        // value (e.g. 0 for sum, 1 for product); reducing along a different
        // axis preserves the zero in the output. Both cases are produced by
        // splatting the identity at `new_shape` — the kernel would otherwise
        // panic on `iterations > 0` in `tile-ir/.../reduce.rs`.
        if input_shape.contains(&0) {
            let data =
                TensorData::new_splat_scalar(&device, &new_shape, function.function.initial_value);
            return Self::new(data);
        }
        info = TensorInfo::new(new_shape, info.datatype());
        let key = device.compute_graph().create_reduce(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn rms_norm(&self, function: RmsNormOperation) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let key = device.compute_graph().create_rms_norm(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn flash_attention(&self, function: FlashAttentionOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = function.out_shape.clone();
        let key = device.compute_graph().create_flash_attention(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn map_layout(&self, op: MapLayoutOperation) -> Self {
        let device = self.device.clone();
        // Compute output shape by applying the layout transformation to a temporary layout
        let temp_layout = Layout::contiguous(self.info.shape());
        let new_layout = op.map_layout(&temp_layout);
        let info = TensorInfo::new(new_layout.shape().into(), self.info.datatype());
        let key = device.compute_graph().create_map_layout(op);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn resize(&self, op: ResizeOperation) -> Self {
        let device = self.device.clone();
        let info = TensorInfo::new(op.new_shape.clone(), self.info.datatype());
        let key = device.compute_graph().create_resize(op);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn slice_assign(&self, op: SliceAssignOperation) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let key = device.compute_graph().create_slice_assign(op);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn materialize(&self) -> (TensorData, usize) {
        let result = self.device.compute_graph().resolve(self.key);
        (result.data, result.total_kernels)
    }

    #[cfg(feature = "graphvis")]
    pub fn graphvis(&self) -> Graph {
        self.device.compute_graph().graphvis(self.key)
    }
}
