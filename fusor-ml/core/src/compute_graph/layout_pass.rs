use rustc_hash::FxHashMap;

use crate::{Layout, TensorLayoutInfo, nary_wise::ElementwiseOperation};

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
                ComputeGraphNodeVariant::Tensor(op) => self.visit_tensor(node, op),
                ComputeGraphNodeVariant::QMatrix(op) => self.visit_dequantize(node, op),
                ComputeGraphNodeVariant::Elementwise(op) => self.visit_nary(node, op),
                ComputeGraphNodeVariant::Reduce(op) => self.visit_reduce(node, op),
                ComputeGraphNodeVariant::View(op) => self.visit_view(node, op),
                ComputeGraphNodeVariant::Assign(op) => self.visit_slice_assign(node, op),
            }
        }
    }

    fn visit_nary(&mut self, key: NodeIndex, operation: &ElementwiseOperation) {
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

    fn visit_reduce(&mut self, key: NodeIndex, operation: &crate::ReduceOperation) {
        let new_layout = Layout::contiguous(&operation.out_shape());
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(new_layout, operation.out_datatype()),
        );
    }

    fn visit_view(&mut self, key: NodeIndex, operation: &crate::view::ViewOperation) {
        let Some(input_layout) = self.output_layout.get(&operation.input) else {
            self.queue.push_back(operation.input);
            self.queue.push_back(key);
            return;
        };
        // A fully-defined view whose stages compose with the input's layout
        // stays a zero-cost buffer view; anything else materializes
        // contiguously through the gather fallback.
        let new_layout = operation
            .is_fully_defined()
            .then(|| {
                operation
                    .stages
                    .iter()
                    .try_fold(input_layout.layout().clone(), |composed, stage| {
                        crate::view::compose_layouts(&stage.layout, &composed)
                    })
            })
            .flatten()
            .unwrap_or_else(|| Layout::contiguous(operation.shape()));
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
        self.output_layout.insert(
            key,
            TensorLayoutInfo::new(
                Layout::contiguous(input_layout.shape()),
                input_layout.datatype(),
            ),
        );
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
}
