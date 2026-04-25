use std::sync::Arc;

use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::CommandEncoder;

use crate::{
    mir::{inputs::MirValue, operation::Operation},
    quantized::matmul::QMatMulOperation,
    tensor::TensorData,
};

use super::{ComputeGraphInner, ComputeGraphNode, ComputeGraphNodeVariant, NodeIndex};

pub(crate) struct ResolverResult {
    pub(crate) data: TensorData,
    pub(crate) total_kernels: usize,
}

#[derive(Debug, Clone)]
struct ExecutionNode {
    inner_idx: NodeIndex,
    variant: ComputeGraphNodeVariant,
}

type ExecutionGraph = StableGraph<ExecutionNode, ()>;
type ExecutionNodeIndex = petgraph::graph::NodeIndex;

pub(crate) struct Resolver {
    execution_graph: ExecutionGraph,
    node_mapping: FxHashMap<NodeIndex, ExecutionNodeIndex>,
    targets: Vec<NodeIndex>,
    resolved_set: FxHashSet<NodeIndex>,
}

impl Resolver {
    pub(crate) fn new(graph: &mut ComputeGraphInner, target: NodeIndex) -> Self {
        Self::new_batch(graph, vec![target])
    }

    pub(crate) fn new_batch(graph: &mut ComputeGraphInner, targets: Vec<NodeIndex>) -> Self {
        let resolved_set = graph
            .nodes
            .nodes
            .node_indices()
            .filter(|&idx| {
                graph
                    .nodes
                    .nodes
                    .node_weight(idx)
                    .map(|n| n.cached.is_some())
                    .unwrap_or(false)
            })
            .collect();
        Self {
            targets,
            execution_graph: Default::default(),
            node_mapping: Default::default(),
            resolved_set,
        }
    }

    pub(crate) fn run(
        &mut self,
        graph: &mut ComputeGraphInner,
        removed: &mut Vec<ComputeGraphNode>,
    ) -> ResolverResult {
        let device = graph.device();

        let targets = self.targets.clone();
        for &target in &targets {
            self.build_execution_graph(graph, target);
        }

        // Tensor IR owns rewrite/fusion now. The resolver only preserves graph
        // semantics, resolves non-kernel layout changes, and executes lowered IR.
        let sorted_nodes = toposort(&self.execution_graph, None)
            .unwrap_or_else(|_| panic!("Cycle detected in execution graph"));

        let target_set: FxHashSet<NodeIndex> = self.targets.iter().copied().collect();
        let mut queued_operations = Vec::with_capacity(sorted_nodes.len());

        for idx in sorted_nodes {
            let node = &self.execution_graph[idx];
            if let ComputeGraphNodeVariant::Tensor(data) = &node.variant {
                graph.set_cached_result(node.inner_idx, data.clone());
                continue;
            }

            if let Some(op) = self.lower_node(node) {
                queued_operations.push((node.inner_idx, op));
            }
        }

        let mut remaining_consumers: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for (_, op) in &queued_operations {
            op.visit_dependencies(&mut |dep| {
                *remaining_consumers.entry(dep).or_insert(0) += 1;
            });
        }

        let mut command_encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Resolver Encoder"),
                });

        let mut total_kernels = 0;
        for (node, operation) in queued_operations {
            let inputs = operation.inputs(graph);
            let map_layout = graph.nodes.nodes.node_weight(node).and_then(|node_data| {
                match &node_data.variant {
                    ComputeGraphNodeVariant::MapLayout(map_layout) => Some(map_layout.clone()),
                    ComputeGraphNodeVariant::Resize(resize) => resize.lower(graph),
                    _ => None,
                }
            });

            if let Some(map_layout) = map_layout {
                let result = map_layout.run(graph);
                graph.set_cached_result(node, result);
                Self::release_dead_intermediates_from_graph(
                    graph,
                    &[node],
                    &mut remaining_consumers,
                    &target_set,
                );
                continue;
            }

            Self::flush_operation(
                graph,
                node,
                &operation,
                &inputs,
                removed,
                &mut command_encoder,
            );
            total_kernels += 1;
            Self::release_dead_intermediates(
                graph,
                &[(node, operation)],
                &mut remaining_consumers,
                &target_set,
            );

            device.wgpu_queue().submit(Some(command_encoder.finish()));
            device.reset_initialized_buffers();
            command_encoder =
                device
                    .wgpu_device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Resolver Encoder"),
                    });
        }

        device.wgpu_queue().submit(Some(command_encoder.finish()));
        device.reset_initialized_buffers();

        let data = graph
            .get_result(self.targets[0])
            .expect("Target result not cached");
        ResolverResult {
            data,
            total_kernels,
        }
    }

    fn release_dead_intermediates(
        graph: &mut ComputeGraphInner,
        produced_ops: &[(NodeIndex, Arc<dyn Operation>)],
        remaining_consumers: &mut FxHashMap<NodeIndex, usize>,
        targets: &FxHashSet<NodeIndex>,
    ) {
        for (_, op) in produced_ops {
            op.visit_dependencies(&mut |dep| {
                if let Some(count) = remaining_consumers.get_mut(&dep) {
                    *count = count.saturating_sub(1);
                    if *count == 0
                        && !targets.contains(&dep)
                        && let Some(node) = graph.nodes.nodes.node_weight_mut(dep)
                    {
                        node.cached = None;
                    }
                }
            });
        }
    }

    fn release_dead_intermediates_from_graph(
        graph: &mut ComputeGraphInner,
        produced_nodes: &[NodeIndex],
        remaining_consumers: &mut FxHashMap<NodeIndex, usize>,
        targets: &FxHashSet<NodeIndex>,
    ) {
        for &produced in produced_nodes {
            let mut deps = Vec::new();
            graph.visit_dependencies(produced, &mut |dep| {
                deps.push(dep);
            });
            for dep in deps {
                if let Some(count) = remaining_consumers.get_mut(&dep) {
                    *count = count.saturating_sub(1);
                    if *count == 0
                        && !targets.contains(&dep)
                        && let Some(node) = graph.nodes.nodes.node_weight_mut(dep)
                    {
                        node.cached = None;
                    }
                }
            }
        }
    }

    fn build_execution_graph(
        &mut self,
        graph: &ComputeGraphInner,
        node: NodeIndex,
    ) -> Option<ExecutionNodeIndex> {
        if self.resolved_set.contains(&node) {
            return None;
        }
        if let Some(&idx) = self.node_mapping.get(&node) {
            return Some(idx);
        }

        let node_data = graph
            .nodes
            .nodes
            .node_weight(node)
            .expect("Node not found in graph");
        let variant = node_data.variant.clone();

        let exec_idx = self.execution_graph.add_node(ExecutionNode {
            inner_idx: node,
            variant: variant.clone(),
        });
        self.node_mapping.insert(node, exec_idx);

        let mut dependencies = Vec::new();
        variant.visit_dependencies(&mut |dependency| {
            dependencies.push(dependency);
        });

        for dependency in dependencies {
            if let Some(dep_exec_idx) = self.build_execution_graph(graph, dependency) {
                self.execution_graph.add_edge(dep_exec_idx, exec_idx, ());
            }
        }

        Some(exec_idx)
    }

    fn lower_node(&self, node: &ExecutionNode) -> Option<Arc<dyn Operation>> {
        match &node.variant {
            ComputeGraphNodeVariant::Nary(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::MatMul(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::Reduce(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::MapLayout(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::Resize(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::SliceAssign(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::QMatMul(op) => Some(Arc::new(QMatMulOperation::new(
                op.input_datatype,
                &op.in_shape,
                op.input,
                op.matrix.clone(),
            ))),
            ComputeGraphNodeVariant::Dequantize(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::Tensor(_) => None,
            ComputeGraphNodeVariant::Custom(op) => Some(op.clone()),
        }
    }

    fn flush_operation(
        graph: &mut ComputeGraphInner,
        key: NodeIndex,
        operation: &Arc<dyn Operation>,
        inputs: &[MirValue],
        removed: &mut Vec<ComputeGraphNode>,
        command_encoder: &mut CommandEncoder,
    ) {
        let lowered = operation
            .build_tensor_ir(graph, inputs)
            .unwrap_or_else(|error| {
                panic!("failed to lower {} to tensor_ir: {error}", operation.name())
            });
        let device = graph.device();
        let result = crate::tensor_ir_runtime::execute(
            &device,
            &lowered.program,
            &lowered.inputs,
            &lowered.output_shape,
            lowered.output_datatype,
            command_encoder,
        )
        .unwrap_or_else(|error| {
            panic!(
                "failed to execute tensor_ir for {}: {error}",
                operation.name()
            )
        });
        graph.set_cached_result(key, result);

        let mut dependencies = Vec::new();
        graph.visit_dependencies(key, &mut |dependent_key| {
            dependencies.push(dependent_key);
        });
        for dependency in dependencies {
            graph.check_life(dependency, removed);
        }
    }
}
