use std::collections::VecDeque;
use std::fmt::Write;
use std::sync::Arc;

use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use rustc_hash::{FxHashMap, FxHashSet};
use wgpu::CommandEncoder;

use crate::{
    mir::{
        inputs::{KernelInputValue, MirValue},
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{self, WorkgroupShapeConstraints},
    },
    nary_wise::{ExtractedUnaryChain, NaryExpr, NaryOperation, UnaryFunctionChain},
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
        let max_subgroup_size = device.max_subgroup_size();

        // Pass 1: Build execution graph for all targets
        let targets = self.targets.clone();
        for &target in &targets {
            self.build_execution_graph(graph, target);
        }

        // Pass 2: Apply Rewrite Rules
        self.optimize(graph);

        // Pass 3: Topological Sort
        let sorted_nodes = toposort(&self.execution_graph, None)
            .unwrap_or_else(|_| panic!("Cycle detected in execution graph"));

        // Pass 4: Execution
        // Extract operations in order.
        let target_set: FxHashSet<NodeIndex> = self.targets.iter().copied().collect();
        let mut queued_operations = Vec::with_capacity(sorted_nodes.len());

        for idx in sorted_nodes {
            let node = &self.execution_graph[idx];
            // Handle Tensor caching explicitly here
            if let ComputeGraphNodeVariant::Tensor(data) = &node.variant {
                graph.set_cached_result(node.inner_idx, data.clone());
                continue;
            }

            if let Some(op) = self.lower_node(node) {
                queued_operations.push((node.inner_idx, op));
            }
        }

        // Build a remaining-consumer count. For each queued operation, we use
        // the Operation's visit_dependencies (which reflects post-optimization
        // fused dependencies) to count how many future operations read each
        // inner NodeIndex.
        let mut remaining_consumers: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        for (_, op) in &queued_operations {
            op.visit_dependencies(&mut |dep| {
                *remaining_consumers.entry(dep).or_insert(0) += 1;
            });
        }

        // Create the first command encoder. We submit and recreate it after
        // each kernel flush so that the GPU can reclaim freed intermediate
        // buffers instead of accumulating all commands in one giant encoder.
        let mut command_encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Resolver Encoder"),
                });

        // Find runs of compatible dispatch shapes
        let mut current_constraints = WorkgroupShapeConstraints::new();
        let mut pending_operations = Vec::new();
        let mut inputs = Vec::new();
        let mut all_input_values = Vec::new();
        let mut kernel = GenericKernel::new();
        let mut total_kernels = 0;
        for (node, operation) in queued_operations {
            let new_inputs = operation.inputs(graph);
            let constraint = operation.workgroup_shape_constraints(&device);
            let mut new_merged = current_constraints.clone();
            new_merged.merge(&constraint);
            let old_best = current_constraints.solve(max_subgroup_size).unwrap_or_else(|| {
                panic!(
                    "Failed to find a valid workgroup shape for constraints {current_constraints:?}"
                )
            });
            let mut extend = self.should_extend_kernel(new_inputs.clone(), &inputs);
            extend &= new_merged.solve(max_subgroup_size).is_some();
            if extend {
                current_constraints = new_merged;
            } else {
                if !pending_operations.is_empty() {
                    total_kernels += 1;
                    Self::flush_operations(
                        graph,
                        &mut kernel,
                        &pending_operations,
                        &inputs,
                        &all_input_values,
                        old_best,
                        removed,
                        &mut command_encoder,
                    );
                    // After flushing, free intermediate cached results whose
                    // consumers within this execution are all satisfied.
                    Self::release_dead_intermediates(
                        graph,
                        &pending_operations,
                        &mut remaining_consumers,
                        &target_set,
                    );
                    // Submit the current command encoder so the GPU can
                    // reclaim buffers we just freed, then start a new one.
                    device.wgpu_queue().submit(Some(command_encoder.finish()));
                    device.reset_initialized_buffers();
                    command_encoder = device.wgpu_device().create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("Resolver Encoder"),
                        },
                    );
                    pending_operations.clear();
                    all_input_values.clear();
                    inputs.clear();
                    kernel.clear();
                }
                current_constraints = constraint;
            }
            // Map layout isn't really a kernel. Resolve it immediately
            let map_layout = if let Some(node_data) = graph.nodes.nodes.node_weight(node) {
                match &node_data.variant {
                    ComputeGraphNodeVariant::MapLayout(map_layout) => Some(map_layout.clone()),
                    ComputeGraphNodeVariant::Resize(resize) => resize.lower(graph),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(map_layout) = map_layout {
                let result = map_layout.run(graph);
                // Cache the result
                graph.set_cached_result(node, result);
                // Map-layout nodes are resolved immediately — release any
                // input buffers that are no longer needed.
                // Use graph.visit_dependencies for map_layout since they
                // are not lowered to Operations.
                Self::release_dead_intermediates_from_graph(
                    graph,
                    &[node],
                    &mut remaining_consumers,
                    &target_set,
                );
            } else {
                self.push_operation(
                    graph,
                    new_inputs,
                    &mut kernel,
                    node,
                    operation,
                    &mut inputs,
                    &mut all_input_values,
                    &mut pending_operations,
                );
            };
        }

        if !pending_operations.is_empty() {
            let old_best = current_constraints.solve(max_subgroup_size).unwrap_or_else(|| {
                panic!(
                    "Failed to find a valid workgroup shape for constraints {current_constraints:?}"
                )
            });
            total_kernels += 1;
            Self::flush_operations(
                graph,
                &mut kernel,
                &pending_operations,
                &inputs,
                &all_input_values,
                old_best,
                removed,
                &mut command_encoder,
            );
            Self::release_dead_intermediates(
                graph,
                &pending_operations,
                &mut remaining_consumers,
                &target_set,
            );
        }

        // Submit any remaining commands.
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

    /// After a kernel flush produces cached results for a set of nodes,
    /// decrement the remaining-consumer count for each of their inputs. When
    /// an input's count reaches zero and it is not a target node, drop its
    /// cached result to free the GPU buffer.
    ///
    /// Uses `op.visit_dependencies()` to match the post-optimization
    /// dependencies that were used to build the consumer counts.
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
                    if *count == 0 && !targets.contains(&dep) {
                        // All consumers within this execution have been
                        // processed — free the cached buffer.
                        if let Some(node) = graph.nodes.nodes.node_weight_mut(dep) {
                            node.cached = None;
                        }
                    }
                }
            });
        }
    }

    /// Like `release_dead_intermediates` but uses the compute graph's
    /// `visit_dependencies` instead of an Operation's. Used for map-layout
    /// and resize nodes that are resolved immediately without being lowered
    /// to an Operation.
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

        // Add to execution graph
        let exec_idx = self.execution_graph.add_node(ExecutionNode {
            inner_idx: node,
            variant: variant.clone(),
        });
        self.node_mapping.insert(node, exec_idx);

        // Find dependencies
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
            ComputeGraphNodeVariant::Tensor(_) => None, // Handled in execution loop
            ComputeGraphNodeVariant::Custom(op) => Some(op.clone()),
        }
    }

    // --- Rewrite Engine ---

    fn optimize(&mut self, graph: &mut ComputeGraphInner) {
        // Initialize worklist with all nodes
        let mut worklist: VecDeque<ExecutionNodeIndex> =
            self.execution_graph.node_indices().collect();
        let mut in_worklist: FxHashSet<ExecutionNodeIndex> = worklist.iter().copied().collect();

        while let Some(node_idx) = worklist.pop_front() {
            in_worklist.remove(&node_idx);

            if !self.execution_graph.contains_node(node_idx) {
                continue;
            }

            // Collect neighbors before optimization (they may need re-processing)
            let neighbors: Vec<_> = self
                .execution_graph
                .neighbors_undirected(node_idx)
                .collect();

            // 1. Fuse naries together (combine expression trees)
            // 2. Try to fuse resulting nary into specialized ops (reduce, matmul, etc.)
            let changed = self.try_fuse_naries(graph, node_idx)
                || self.try_fuse_into_reduce(graph, node_idx)
                || self.try_fuse_into_matmul(graph, node_idx)
                || self.try_fuse_into_dequantize(graph, node_idx);

            if changed {
                // Re-add the current node to worklist if it still exists
                if self.execution_graph.contains_node(node_idx) && !in_worklist.contains(&node_idx)
                {
                    worklist.push_back(node_idx);
                    in_worklist.insert(node_idx);
                }

                // Re-add neighbors that might be affected by this change
                for neighbor in neighbors {
                    if self.execution_graph.contains_node(neighbor)
                        && !in_worklist.contains(&neighbor)
                    {
                        worklist.push_back(neighbor);
                        in_worklist.insert(neighbor);
                    }
                }

                // Also add new neighbors that may have been created
                if self.execution_graph.contains_node(node_idx) {
                    for neighbor in self.execution_graph.neighbors_undirected(node_idx) {
                        if !in_worklist.contains(&neighbor) {
                            worklist.push_back(neighbor);
                            in_worklist.insert(neighbor);
                        }
                    }
                }
            }
        }
    }

    // Helpers
    fn add_physical_dependencies(
        &self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        inputs: &[NodeIndex],
    ) {
        let inner_idx = self.execution_graph[node_idx].inner_idx;
        for &input in inputs {
            graph.nodes.nodes.add_edge(input, inner_idx, ());
        }
    }

    fn get_input_node_in_exec_graph(&self, inner_input: NodeIndex) -> Option<ExecutionNodeIndex> {
        self.node_mapping.get(&inner_input).copied()
    }

    fn check_cached(&self, graph: &ComputeGraphInner, inner_idx: NodeIndex) -> bool {
        graph.get_cached_result(inner_idx).is_some()
    }

    fn remove_node_if_dead(&mut self, node_idx: ExecutionNodeIndex) {
        if !self.execution_graph.contains_node(node_idx) {
            return;
        }
        if self
            .execution_graph
            .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
            .count()
            == 0
        {
            // Collect incoming neighbors before removing
            let incoming: Vec<_> = self
                .execution_graph
                .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                .collect();
            self.execution_graph.remove_node(node_idx);
            // Recursively check if dependencies are now dead
            for dep in incoming {
                self.remove_node_if_dead(dep);
            }
        }
    }

    // Rules

    /// Fuse a Nary operation with all of its Nary inputs.
    fn try_fuse_naries(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        let ComputeGraphNodeVariant::Nary(nary) = node_variant else {
            return false;
        };

        // Collect all fusible nary inputs
        let mut expression = nary.expression.clone();
        let mut all_inputs = nary.inputs.clone();
        let mut fused_execs = Vec::new();

        // Get the max storage buffers limit from GPU
        let max_storage_bindings =
            graph.device().limits().max_storage_buffers_per_shader_stage as usize;

        for (input_idx, &input_inner) in nary.inputs.iter().enumerate() {
            if self.check_cached(graph, input_inner) {
                continue;
            }
            let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) else {
                continue;
            };
            // Check if the node still exists (it may have been removed during optimization)
            if !self.execution_graph.contains_node(input_exec) {
                continue;
            }
            let ComputeGraphNodeVariant::Nary(input_nary) =
                &self.execution_graph[input_exec].variant
            else {
                continue;
            };

            // Inline: offset input nary's indices to append after current inputs
            let offset = all_inputs.len();
            let inlined = Self::offset_input_indices(&input_nary.expression, offset);
            let (new_expression, success) =
                Self::substitute_input_in_expr(&expression, input_idx, &inlined);

            // Only fuse if substitution was successful
            // If not, the expression still references the original input which must remain
            if success {
                // Check if fusing would exceed the GPU's per-stage buffer limit.
                // On Metal, `max_buffers_per_stage` is 31 and covers ALL buffer
                // types (storage + uniform) plus an implicit sizes buffer added
                // by wgpu. Each unique tensor input needs at least 1 storage
                // binding, and there will also be a small number of uniform
                // "info" bindings (for tensor shape/stride metadata), plus the
                // output storage binding, plus the wgpu sizes buffer.
                //
                // We use `max_storage_bindings` (which equals `max_buffers_per_stage`
                // on Metal = 31) as the hard ceiling and reserve slots for:
                //   - 1 output storage binding
                //   - 1 wgpu sizes buffer
                //   - up to `info_headroom` uniform info bindings
                let info_headroom = 4usize;
                let max_fused_inputs = max_storage_bindings.saturating_sub(2 + info_headroom);

                // Count unique inputs after potential merge (duplicates share a binding)
                let unique_inputs: FxHashSet<_> = all_inputs
                    .iter()
                    .chain(input_nary.inputs.iter())
                    .copied()
                    .collect();

                if unique_inputs.len() >= max_fused_inputs {
                    // Skip fusion - would exceed GPU binding limit
                    continue;
                }

                expression = new_expression;
                all_inputs.extend(input_nary.inputs.iter().copied());
                fused_execs.push((input_exec, input_nary.inputs.clone()));
            }
        }

        if fused_execs.is_empty() {
            return false;
        }

        // Deduplicate and remove unused inputs
        let (final_inputs, final_expression) = Self::deduplicate_inputs(all_inputs, expression);

        let new_nary = NaryOperation {
            inputs: final_inputs.clone(),
            expression: final_expression,
            shape: nary.shape.clone(),
            output_datatype: nary.output_datatype,
        };

        self.execution_graph[node_idx].variant = ComputeGraphNodeVariant::Nary(new_nary.clone());

        // Update graph edges
        for (input_exec, new_inputs) in fused_execs {
            if let Some(edge) = self.execution_graph.find_edge(input_exec, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
            for &new_input in &new_inputs {
                if let Some(exec) = self.get_input_node_in_exec_graph(new_input)
                    && self.execution_graph.find_edge(exec, node_idx).is_none()
                {
                    self.execution_graph.add_edge(exec, node_idx, ());
                }
            }
            self.remove_node_if_dead(input_exec);
        }

        self.add_physical_dependencies(graph, node_idx, &new_nary.inputs);
        true
    }

    /// Add offset to all input indices in an expression.
    fn offset_input_indices(expr: &NaryExpr, offset: usize) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|c| Self::offset_input_indices(c, offset))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => NaryExpr::IndexedInput {
                input_idx: input_idx + offset,
                indices: indices
                    .iter()
                    .map(|c| Self::offset_input_indices(c, offset))
                    .collect(),
            },
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
        }
    }

    /// Substitute IndexedInput(target_idx) with element-wise access with the replacement expression.
    /// Returns (new_expression, success) where success is true if all references to target_idx
    /// were successfully substituted. If false, the input should NOT be removed from the graph.
    fn substitute_input_in_expr(
        expr: &NaryExpr,
        target_idx: usize,
        replacement: &NaryExpr,
    ) -> (NaryExpr, bool) {
        /// Helper to extract input_idx from an IndexedInput with element-wise access
        fn get_elementwise_input_idx(expr: &NaryExpr) -> Option<usize> {
            match expr {
                NaryExpr::IndexedInput { input_idx, indices }
                    if NaryExpr::is_elementwise_indices(indices) =>
                {
                    Some(*input_idx)
                }
                _ => None,
            }
        }

        match expr {
            NaryExpr::Op { children, function } => {
                let mut all_success = true;
                let new_children: Vec<_> = children
                    .iter()
                    .map(|c| {
                        let (new_c, success) =
                            Self::substitute_input_in_expr(c, target_idx, replacement);
                        all_success &= success;
                        new_c
                    })
                    .collect();
                (
                    NaryExpr::Op {
                        children: new_children,
                        function: function.clone(),
                    },
                    all_success,
                )
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                if *input_idx == target_idx {
                    // Check if this is element-wise access
                    if NaryExpr::is_elementwise_indices(indices) {
                        // Element-wise can be fully replaced with any expression
                        (replacement.clone(), true)
                    } else {
                        // Custom indexing can only substitute if replacement is also element-wise
                        if let Some(new_idx) = get_elementwise_input_idx(replacement) {
                            let mut all_success = true;
                            let new_indices: Vec<_> = indices
                                .iter()
                                .map(|c| {
                                    let (new_c, success) =
                                        Self::substitute_input_in_expr(c, target_idx, replacement);
                                    all_success &= success;
                                    new_c
                                })
                                .collect();
                            (
                                NaryExpr::IndexedInput {
                                    input_idx: new_idx,
                                    indices: new_indices,
                                },
                                all_success,
                            )
                        } else {
                            // Cannot fuse complex expression into custom indexed input
                            let all_success = false;
                            let new_indices: Vec<_> = indices
                                .iter()
                                .map(|c| {
                                    let (new_c, _) =
                                        Self::substitute_input_in_expr(c, target_idx, replacement);
                                    new_c
                                })
                                .collect();
                            (
                                NaryExpr::IndexedInput {
                                    input_idx: *input_idx,
                                    indices: new_indices,
                                },
                                all_success,
                            )
                        }
                    }
                } else {
                    // Recurse into the index expressions
                    let mut all_success = true;
                    let new_indices: Vec<_> = indices
                        .iter()
                        .map(|c| {
                            let (new_c, s) =
                                Self::substitute_input_in_expr(c, target_idx, replacement);
                            all_success &= s;
                            new_c
                        })
                        .collect();
                    (
                        NaryExpr::IndexedInput {
                            input_idx: *input_idx,
                            indices: new_indices,
                        },
                        all_success,
                    )
                }
            }
            NaryExpr::DimIndex(dim) => (NaryExpr::DimIndex(*dim), true),
        }
    }

    /// Remove unused inputs and deduplicate, returning new inputs and remapped expression.
    fn deduplicate_inputs(inputs: Vec<NodeIndex>, expr: NaryExpr) -> (Vec<NodeIndex>, NaryExpr) {
        // Collect which input indices are actually used
        let mut used_indices = FxHashSet::default();
        Self::collect_used_inputs(&expr, &mut used_indices);

        // Build mapping: old index -> new index, and collect only used inputs
        let mut new_inputs = Vec::new();
        let mut old_to_new = FxHashMap::default();

        for old_idx in used_indices.iter().copied().collect::<Vec<_>>() {
            let node = inputs[old_idx];
            // Check if this node already exists in new_inputs (deduplication)
            let new_idx = if let Some(existing) = new_inputs.iter().position(|&n| n == node) {
                existing
            } else {
                let idx = new_inputs.len();
                new_inputs.push(node);
                idx
            };
            old_to_new.insert(old_idx, new_idx);
        }

        let new_expr = Self::remap_input_indices(&expr, &old_to_new);
        (new_inputs, new_expr)
    }

    fn collect_used_inputs(expr: &NaryExpr, used: &mut FxHashSet<usize>) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    Self::collect_used_inputs(child, used);
                }
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                used.insert(*input_idx);
                for c in indices {
                    Self::collect_used_inputs(c, used);
                }
            }
            NaryExpr::DimIndex(_) => {}
        }
    }

    fn remap_input_indices(expr: &NaryExpr, mapping: &FxHashMap<usize, usize>) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|c| Self::remap_input_indices(c, mapping))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => NaryExpr::IndexedInput {
                input_idx: mapping[input_idx],
                indices: indices
                    .iter()
                    .map(|c| Self::remap_input_indices(c, mapping))
                    .collect(),
            },
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
        }
    }

    /// Try to extract a unary function chain from a node variant.
    /// Only Nary ops with a single input and element-wise access can be converted.
    fn try_get_unary_chain(variant: &ComputeGraphNodeVariant) -> Option<ExtractedUnaryChain> {
        match variant {
            ComputeGraphNodeVariant::Nary(nary) => nary.try_extract_unary_chain(),
            _ => None,
        }
    }

    fn try_fuse_into_reduce(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        let Some(el_op) = Self::try_get_unary_chain(&node_variant) else {
            return false;
        };

        let input_inner = el_op.value;
        if self.check_cached(graph, input_inner) {
            return false;
        }

        let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner) else {
            return false;
        };

        let input_variant = self.execution_graph[input_exec_idx].variant.clone();
        let ComputeGraphNodeVariant::Reduce(reduce_op) = input_variant else {
            return false;
        };

        let mut new_reduce = reduce_op.clone();
        let mut existing_post = new_reduce.post_element_wise.functions.clone();
        existing_post.extend(el_op.functions.functions.iter().cloned());
        new_reduce.post_element_wise =
            UnaryFunctionChain::new(existing_post, reduce_op.post_element_wise.input_datatype());

        self.execution_graph[node_idx].variant =
            ComputeGraphNodeVariant::Reduce(new_reduce.clone());

        let reduce_input_inner = reduce_op.value;
        if let Some(reduce_input_exec) = self.get_input_node_in_exec_graph(reduce_input_inner) {
            self.execution_graph
                .add_edge(reduce_input_exec, node_idx, ());
        }

        if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
            self.execution_graph.remove_edge(edge);
        }
        self.add_physical_dependencies(graph, node_idx, &[reduce_input_inner]);
        self.remove_node_if_dead(input_exec_idx);
        true
    }

    fn try_fuse_into_matmul(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        // Post-op: fuse elementwise after matmul
        if let Some(el_op) = Self::try_get_unary_chain(&node_variant) {
            let input_inner = el_op.value;
            if !self.check_cached(graph, input_inner)
                && let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner)
            {
                let input_variant = self.execution_graph[input_exec_idx].variant.clone();
                if let ComputeGraphNodeVariant::MatMul(matmul_op) = input_variant {
                    let mut new_matmul = matmul_op.clone();
                    let mut existing_post = new_matmul.post_element_wise.functions.clone();
                    existing_post.extend(el_op.functions.functions.iter().cloned());
                    new_matmul.post_element_wise = UnaryFunctionChain::new(
                        existing_post,
                        matmul_op.post_element_wise.input_datatype(),
                    );

                    self.execution_graph[node_idx].variant =
                        ComputeGraphNodeVariant::MatMul(new_matmul.clone());

                    let (first_inner, second_inner) = (matmul_op.first, matmul_op.second);
                    if let Some(idx) = self.get_input_node_in_exec_graph(first_inner) {
                        self.execution_graph.add_edge(idx, node_idx, ());
                    }
                    if let Some(idx) = self.get_input_node_in_exec_graph(second_inner) {
                        self.execution_graph.add_edge(idx, node_idx, ());
                    }
                    if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    self.add_physical_dependencies(graph, node_idx, &[first_inner, second_inner]);
                    self.remove_node_if_dead(input_exec_idx);
                    return true;
                }
            }
        }

        // Pre-op: fuse elementwise before matmul inputs
        if let ComputeGraphNodeVariant::MatMul(matmul_op) = &node_variant {
            let mut new_matmul = matmul_op.clone();
            let mut changed = false;

            // Check first input
            if !self.check_cached(graph, matmul_op.first)
                && let Some(first_exec) = self.get_input_node_in_exec_graph(matmul_op.first)
                && let Some(el_op) =
                    Self::try_get_unary_chain(&self.execution_graph[first_exec].variant)
            {
                new_matmul.first = el_op.value;
                new_matmul.pre_element_wise[0] = el_op.functions.clone();
                changed = true;
            }

            // Check second input
            if !self.check_cached(graph, matmul_op.second)
                && let Some(second_exec) = self.get_input_node_in_exec_graph(matmul_op.second)
                && let Some(el_op) =
                    Self::try_get_unary_chain(&self.execution_graph[second_exec].variant)
            {
                new_matmul.second = el_op.value;
                new_matmul.pre_element_wise[1] = el_op.functions.clone();
                changed = true;
            }

            if changed {
                self.execution_graph[node_idx].variant =
                    ComputeGraphNodeVariant::MatMul(new_matmul.clone());

                if new_matmul.first != matmul_op.first {
                    let old = self.get_input_node_in_exec_graph(matmul_op.first).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new) = self.get_input_node_in_exec_graph(new_matmul.first) {
                        self.execution_graph.add_edge(new, node_idx, ());
                    }
                    self.remove_node_if_dead(old);
                }
                if new_matmul.second != matmul_op.second {
                    let old = self.get_input_node_in_exec_graph(matmul_op.second).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new) = self.get_input_node_in_exec_graph(new_matmul.second) {
                        self.execution_graph.add_edge(new, node_idx, ());
                    }
                    self.remove_node_if_dead(old);
                }
                self.add_physical_dependencies(
                    graph,
                    node_idx,
                    &[new_matmul.first, new_matmul.second],
                );
                return true;
            }
        }

        false
    }

    fn try_fuse_into_dequantize(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();
        let Some(el_op) = Self::try_get_unary_chain(&node_variant) else {
            return false;
        };

        let input_inner = el_op.value;
        if self.check_cached(graph, input_inner) {
            return false;
        }

        let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner) else {
            return false;
        };

        let input_variant = self.execution_graph[input_exec_idx].variant.clone();
        let ComputeGraphNodeVariant::Dequantize(deq_op) = input_variant else {
            return false;
        };

        let mut new_deq = deq_op.clone();
        let mut existing_post = new_deq.post_dequantize.functions.clone();
        existing_post.extend(el_op.functions.functions.iter().cloned());
        new_deq.post_dequantize = UnaryFunctionChain::new(existing_post, deq_op.datatype);

        self.execution_graph[node_idx].variant = ComputeGraphNodeVariant::Dequantize(new_deq);

        if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
            self.execution_graph.remove_edge(edge);
        }
        self.remove_node_if_dead(input_exec_idx);
        true
    }

    fn should_extend_kernel(&mut self, _: Vec<MirValue>, _: &[Vec<MirValue>]) -> bool {
        // TODO: Restore with better testing. This passes all tests in fusor, but breaks rbert and rwhisper
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn push_operation(
        &mut self,
        graph: &mut ComputeGraphInner,
        new_inputs: Vec<MirValue>,
        kernel: &mut GenericKernel,
        key: NodeIndex,
        operation: Arc<dyn Operation>,
        inputs: &mut Vec<Vec<MirValue>>,
        all_input_values: &mut Vec<KernelInputValue>,
        queued_operations: &mut Vec<(NodeIndex, Arc<dyn Operation>)>,
    ) {
        for input in &new_inputs {
            input.visit_input_values(|value| {
                if let Some(index) = all_input_values.iter().position(|x| *x == value) {
                    kernel.pre_register_binding(index as _);
                } else {
                    kernel.pre_register_binding(all_input_values.len() as _);
                    all_input_values.push(value.clone());
                }
            });
        }
        let result = operation.output(graph, &new_inputs);
        let MirValue::Tensor(resolved) = result else {
            panic!("Kernel input value is not a tensor");
        };
        // Cache the result
        graph.set_cached_result(key, resolved);
        inputs.push(new_inputs);
        queued_operations.push((key, operation));
    }

    fn flush_operations(
        graph: &mut ComputeGraphInner,
        mut kernel: &mut GenericKernel,
        queued_operations: &[(NodeIndex, Arc<dyn Operation>)],
        inputs: &[Vec<MirValue>],
        all_input_values: &[KernelInputValue],
        workgroup_shape: workgroup_shape::WorkgroupShape,
        removed: &mut Vec<ComputeGraphNode>,
        command_encoder: &mut CommandEncoder,
    ) {
        let mut max_dispatch_size = [0; 3];
        for ((key, operation), inputs) in queued_operations.iter().zip(inputs) {
            // Map layout isn't really a kernel. Skip it
            if let Some(node) = graph.nodes.nodes.node_weight(*key)
                && matches!(node.variant, ComputeGraphNodeVariant::MapLayout(_))
            {
                continue;
            }

            let dispatch_size = operation.dispatch_size(&workgroup_shape, inputs);
            for (new, max) in dispatch_size.iter().zip(max_dispatch_size.iter_mut()) {
                *max = (*max).max(*new);
            }
            writeln!(&mut kernel, "{{").unwrap();
            operation.build_kernel(graph, &workgroup_shape, inputs, kernel);
            let name = kernel.name_mut();
            if !name.is_empty() {
                *name += "->";
            }
            *name += &operation.name();
            writeln!(&mut kernel, "}}").unwrap();
            // Check if that makes any of this node's dependencies dead
            let mut dependencies = Vec::new();
            graph.visit_dependencies(*key, &mut |dependent_key| {
                dependencies.push(dependent_key);
            });
            for dependency in dependencies {
                graph.check_life(dependency, removed);
            }
        }
        kernel.set_workgroup_size(workgroup_shape);
        let device = graph.device();
        kernel.run(
            &device,
            all_input_values,
            command_encoder,
            max_dispatch_size,
        );
    }
}
