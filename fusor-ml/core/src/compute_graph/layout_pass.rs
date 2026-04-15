use rustc_hash::FxHashMap;

use crate::{Layout, TensorLayoutInfo, nary_wise::NaryOperation};

use super::{ComputeGraphNodeVariant, NodeIndex, queue::ComputeQueue};

#[derive(Default)]
pub(crate) struct LayoutPass {
    queue: ComputeQueue,
    pub(crate) output_layout: FxHashMap<NodeIndex, TensorLayoutInfo>,
}

impl LayoutPass {
    pub fn visit(&mut self, graph: &super::ComputeGraphInner, key: NodeIndex) {
        self.queue.push_back(key);

        while let Some(node) = self.queue.pop_front() {
            if self.output_layout.contains_key(&node) {
                continue;
            }
            let node_data = graph.nodes.nodes.node_weight(node).expect("Node not found");
            if let Some(resolved) = &node_data.cached {
                self.output_layout.insert(node, resolved.info().clone());
                continue;
            }
            match &node_data.variant {
                ComputeGraphNodeVariant::Nary(op) => self.visit_nary(node, op),
                ComputeGraphNodeVariant::MatMul(op) => self.visit_mat_mul(node, op),
                ComputeGraphNodeVariant::QMatMul(op) => self.visit_q_mat_mul(node, op),
                ComputeGraphNodeVariant::Reduce(op) => self.visit_reduce(node, op),
                ComputeGraphNodeVariant::MapLayout(op) => self.visit_map_layout(node, op),
                ComputeGraphNodeVariant::Resize(op) => self.visit_resize(node, op),
                ComputeGraphNodeVariant::SliceAssign(op) => self.visit_slice_assign(node, op),
                ComputeGraphNodeVariant::Tensor(op) => self.visit_tensor(node, op),
                ComputeGraphNodeVariant::Dequantize(op) => self.visit_dequantize(node, op),
                ComputeGraphNodeVariant::Custom(op) => self.visit_custom(node, op),
            }
        }
    }

    fn visit_nary(&mut self, key: NodeIndex, operation: &NaryOperation) {
        // Ensure all inputs have been visited
        for input in &operation.inputs {
            if !self.output_layout.contains_key(input) {
                self.queue.push_back(*input);
                self.queue.push_back(key);
                return;
            }
        }
        let output_layout = Layout::contiguous(&operation.shape);
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(output_layout, operation.output_datatype),
        );
    }

    fn visit_mat_mul(&mut self, key: NodeIndex, operation: &crate::MatMulOperation) {
        let Some(first_layout) = self.output_layout.get(&operation.first) else {
            self.queue.push_back(operation.first);
            self.queue.push_back(key);
            return;
        };
        let Some(_) = self.output_layout.get(&operation.second) else {
            self.queue.push_back(operation.second);
            self.queue.push_back(key);
            return;
        };
        let output_shape = &operation.out_shape;
        let output_layout = Layout::contiguous(output_shape);
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(output_layout, first_layout.datatype()),
        );
    }

    fn visit_q_mat_mul(
        &mut self,
        key: NodeIndex,
        operation: &crate::quantized::matmul::QMatMulOperation,
    ) {
        let Some(first_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        let output_layout = Layout::contiguous(&operation.out_shape);
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(output_layout, first_layout.datatype()),
        );
    }

    fn visit_reduce(&mut self, key: NodeIndex, operation: &crate::ReduceOperation) {
        let dim = operation.axis;
        let Some(input_layout) = self.output_layout.get(&operation.value) else {
            self.queue.push_back(operation.value);
            self.queue.push_back(key);
            return;
        };
        let new_shape = input_layout
            .layout()
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect::<Vec<_>>();
        let new_layout = Layout::contiguous(&new_shape);
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_map_layout(
        &mut self,
        key: NodeIndex,
        operation: &crate::map_layout::MapLayoutOperation,
    ) {
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        let new_layout = operation.map_layout(input_layout.layout());
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_resize(&mut self, key: NodeIndex, operation: &crate::resize::ResizeOperation) {
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        let new_layout = Layout::contiguous(&operation.new_shape);
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(new_layout, input_layout.datatype()),
        );
    }

    fn visit_slice_assign(
        &mut self,
        key: NodeIndex,
        operation: &crate::slice_assign::SliceAssignOperation,
    ) {
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        let Some(_) = self.output_layout.get(&operation.value) else {
            self.queue.push_back(operation.value);
            self.queue.push_back(key);
            return;
        };
        self.output_layout.insert(key, input_layout.clone());
    }

    fn visit_tensor(&mut self, key: NodeIndex, operation: &crate::tensor::TensorData) {
        let info = operation.info();
        self.output_layout.insert(key, info.clone());
    }

    fn visit_dequantize(
        &mut self,
        key: NodeIndex,
        operation: &crate::dequantize::DequantizeOperation,
    ) {
        let matrix = &operation.matrix;
        let new_layout = Layout::contiguous(matrix.shape());
        self.output_layout
            .insert(key, TensorLayoutInfo::new(new_layout, operation.datatype));
    }

    fn visit_custom(
        &mut self,
        key: NodeIndex,
        operation: &std::sync::Arc<dyn crate::mir::operation::Operation + Send + Sync>,
    ) {
        let mut dependencies = Vec::new();
        operation.visit_dependencies(&mut |dep| {
            dependencies.push(dep);
        });

        for dependency in dependencies {
            if !self.output_layout.contains_key(&dependency) {
                self.queue.push_back(dependency);
                self.queue.push_back(key);
                return;
            }
        }
        self.output_layout
            .insert(key, operation.output_layout(&self.output_layout));
    }
}
