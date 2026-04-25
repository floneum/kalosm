use rustc_hash::FxHashMap;

use crate::{TensorLayoutInfo, mir::operation::Operation};

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
                ComputeGraphNodeVariant::TensorExpr(op) => self.visit_tensor_expr(node, op),
                ComputeGraphNodeVariant::MapLayout(op) => self.visit_map_layout(node, op),
                ComputeGraphNodeVariant::Tensor(op) => self.visit_tensor(node, op),
                ComputeGraphNodeVariant::Custom(op) => self.visit_custom(node, op),
            }
        }
    }

    fn visit_tensor_expr(&mut self, key: NodeIndex, operation: &crate::TensorExprOperation) {
        // Ensure all inputs have been visited
        let mut dependencies = Vec::new();
        operation.visit_dependencies(&mut |dep| {
            dependencies.push(dep);
        });
        for input in dependencies {
            if !self.output_layout.contains_key(&input) {
                self.queue.push_back(input);
                self.queue.push_back(key);
                return;
            }
        }
        self.output_layout
            .insert(key, operation.output_layout(&self.output_layout));
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

    fn visit_tensor(&mut self, key: NodeIndex, operation: &crate::tensor::TensorData) {
        let info = operation.info();
        self.output_layout.insert(key, info.clone());
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
