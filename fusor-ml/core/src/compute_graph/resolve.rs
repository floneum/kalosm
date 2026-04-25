use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::CommandEncoder;

use crate::{
    mir::{inputs::MirValue, operation::Operation},
    tensor::TensorData,
};

use super::{ComputeGraphInner, ComputeGraphNode, ComputeGraphNodeVariant, NodeIndex};

pub(crate) struct ResolverResult {
    pub(crate) data: TensorData,
    pub(crate) total_kernels: usize,
}

pub(crate) struct Resolver {
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
            resolved_set,
        }
    }

    pub(crate) fn run(
        &mut self,
        graph: &mut ComputeGraphInner,
        removed: &mut Vec<ComputeGraphNode>,
    ) -> ResolverResult {
        let device = graph.device();

        let mut sorted_nodes = Vec::new();
        let mut visiting = FxHashSet::default();
        let mut visited = FxHashSet::default();
        let targets = self.targets.clone();
        for &target in &targets {
            self.visit_execution_order(
                graph,
                target,
                &mut visiting,
                &mut visited,
                &mut sorted_nodes,
            );
        }

        // Tensor IR owns rewrite/fusion now. The resolver only preserves graph
        // semantics, resolves non-kernel layout changes, and executes lowered IR.
        let target_set: FxHashSet<NodeIndex> = self.targets.iter().copied().collect();
        let mut queued_operations = Vec::with_capacity(sorted_nodes.len());

        for node in sorted_nodes {
            let variant = graph
                .nodes
                .nodes
                .node_weight(node)
                .expect("Node not found in graph")
                .variant
                .clone();
            if let ComputeGraphNodeVariant::Tensor(data) = variant {
                graph.set_cached_result(node, data);
                continue;
            }

            if let Some(op) = Self::lower_variant(&variant) {
                queued_operations.push((node, op));
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
                    ComputeGraphNodeVariant::Custom(op) => op.try_metadata_lower(graph),
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

    fn visit_execution_order(
        &mut self,
        graph: &ComputeGraphInner,
        node: NodeIndex,
        visiting: &mut FxHashSet<NodeIndex>,
        visited: &mut FxHashSet<NodeIndex>,
        sorted: &mut Vec<NodeIndex>,
    ) {
        if self.resolved_set.contains(&node) {
            return;
        }
        if visiting.contains(&node) {
            panic!("Cycle detected in execution graph");
        }
        if !visited.insert(node) {
            return;
        }
        visiting.insert(node);

        let node_data = graph
            .nodes
            .nodes
            .node_weight(node)
            .expect("Node not found in graph");

        let mut dependencies = Vec::new();
        node_data.variant.visit_dependencies(&mut |dependency| {
            dependencies.push(dependency);
        });

        for dependency in dependencies {
            self.visit_execution_order(graph, dependency, visiting, visited, sorted);
        }

        visiting.remove(&node);
        sorted.push(node);
    }

    fn lower_variant(variant: &ComputeGraphNodeVariant) -> Option<Arc<dyn Operation>> {
        match variant {
            ComputeGraphNodeVariant::MapLayout(op) => Some(Arc::new(op.clone())),
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
